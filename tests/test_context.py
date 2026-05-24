import json
import inspect
import subprocess

from coding_agent.context.context import (
    Context,
    ContextFrame,
    UsageStats,
    WorkspaceContextOptions,
    WorkspaceSnapshot,
    estimate_tokens,
)
from coding_agent.context.assembler import ContextAssembler
from coding_agent.context.compressor import parse_compaction_response
from coding_agent.memory.store import MemoryStore


def test_usage_stats_parses_openai_cached_tokens():
    stats = UsageStats.from_response_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "prompt_tokens_details": {"cached_tokens": 85},
        }
    )

    assert stats == UsageStats(input_tokens=100, output_tokens=25, cached_tokens=85)
    assert stats.cache_hit_ratio == 0.85


def test_usage_stats_parses_nested_attribute_cached_tokens():
    class PromptTokenDetails:
        cached_tokens = 85

    class ResponseUsage:
        prompt_tokens = 100
        completion_tokens = 25
        prompt_tokens_details = PromptTokenDetails()

    stats = UsageStats.from_response_usage(ResponseUsage())

    assert stats == UsageStats(input_tokens=100, output_tokens=25, cached_tokens=85)
    assert stats.cache_hit_ratio == 0.85


def test_usage_stats_parses_generic_input_output_shape():
    stats = UsageStats.from_response_usage(
        {"input_tokens": 50, "output_tokens": 10, "cached_tokens": 20}
    )

    assert stats == UsageStats(input_tokens=50, output_tokens=10, cached_tokens=20)
    assert stats.cache_hit_ratio == 0.4


def test_usage_stats_cache_hit_ratio_is_none_without_input_or_cached_tokens():
    assert UsageStats(input_tokens=0, output_tokens=10, cached_tokens=5).cache_hit_ratio is None
    assert UsageStats(input_tokens=10, output_tokens=10, cached_tokens=None).cache_hit_ratio is None
    assert UsageStats.from_response_usage(None) is None


def test_workspace_snapshot_builds_outside_git_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    (tmp_path / "README.md").write_text("hello project", encoding="utf-8")

    snapshot = WorkspaceSnapshot.build(tmp_path, WorkspaceContextOptions())

    assert snapshot.cwd == tmp_path.resolve()
    assert snapshot.repo_root is None
    assert snapshot.branch is None
    assert snapshot.status == ""
    assert snapshot.recent_commits == []
    assert snapshot.project_docs == {"README.md": "hello project"}


def test_workspace_snapshot_clips_project_docs(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    (tmp_path / "README.md").write_text("x" * 20, encoding="utf-8")

    snapshot = WorkspaceSnapshot.build(tmp_path, WorkspaceContextOptions(doc_max_chars=5))

    assert snapshot.project_docs["README.md"] == "xxxxx\n... [truncated]"


def test_workspace_snapshot_file_tree_ignores_generated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "app.pyc").write_text("", encoding="utf-8")
    (tmp_path / ".coding-agent").mkdir()
    (tmp_path / ".coding-agent" / "config.toml").write_text("", encoding="utf-8")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "README.md").write_text("", encoding="utf-8")

    snapshot = WorkspaceSnapshot.build(tmp_path, WorkspaceContextOptions())

    assert "src/app.py" in snapshot.file_tree
    assert "__pycache__/app.pyc" not in snapshot.file_tree
    assert ".coding-agent/config.toml" not in snapshot.file_tree
    assert ".pytest_cache/README.md" not in snapshot.file_tree


def test_workspace_snapshot_collects_git_state(tmp_path):
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "dirty.txt").write_text("dirty", encoding="utf-8")

    snapshot = WorkspaceSnapshot.build(tmp_path, WorkspaceContextOptions())

    assert snapshot.repo_root == tmp_path.resolve()
    assert snapshot.branch == "main"
    assert "?? dirty.txt" in snapshot.status
    assert any("initial" in commit for commit in snapshot.recent_commits)


def test_workspace_snapshot_fingerprint_changes_when_git_status_changes(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    clean_fingerprint = WorkspaceSnapshot.build(tmp_path, WorkspaceContextOptions()).fingerprint()
    (tmp_path / "dirty.txt").write_text("dirty", encoding="utf-8")
    dirty_fingerprint = WorkspaceSnapshot.build(tmp_path, WorkspaceContextOptions()).fingerprint()

    assert dirty_fingerprint != clean_fingerprint


def test_estimate_tokens_uses_ceil_len_div_4_with_minimum_one():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a") == 1
    assert estimate_tokens("a" * 4) == 1
    assert estimate_tokens("a" * 5) == 2


def test_context_add_message_preserves_raw_stream_and_returns_active_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(include_project_docs=False, include_file_tree=False),
        task="do it",
    )
    message = {"role": "user", "content": "hello", "metadata": {"n": 1}}

    context.add_message(message)
    message["metadata"]["n"] = 2
    frames = context.history_frames()
    frames[0].payload["metadata"]["n"] = 3

    assert context.raw_messages() == [{"role": "user", "content": "hello", "metadata": {"n": 1}}]
    assert context.history_frames()[0].payload["metadata"]["n"] == 1


def test_context_constructor_does_not_accept_memory_or_compact_client_dependencies():
    signature = inspect.signature(Context)

    assert "memory_store" not in signature.parameters
    assert "compact_client" not in signature.parameters


def test_update_workspace_snapshot_stores_structured_payload_without_markup(tmp_path):
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(include_project_docs=False, include_file_tree=False),
        task="do it",
    )
    snapshot = WorkspaceSnapshot(
        cwd=tmp_path,
        repo_root=None,
        branch=None,
        default_branch=None,
        status="",
        recent_commits=[],
        project_docs={"README.md": "literal <workspace_context>"},
        file_tree=["README.md"],
    )

    context.update_workspace_snapshot(snapshot)

    workspace_frame = next(frame for frame in context.frames() if frame.kind == "workspace")
    assert workspace_frame.payload["project_docs"]["README.md"] == "literal <workspace_context>"
    assert workspace_frame.kind == "workspace"
    assert "content" not in workspace_frame.payload


def test_context_assembler_pushes_memory_as_raw_structured_data(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    store = MemoryStore(tmp_path)
    store.save_scratchpad({"project_goal": "keep raw", "modified_files": ["src/a.py"]})
    store.write_handoff("handoff text")

    context = ContextAssembler(
        cwd=tmp_path,
        options=WorkspaceContextOptions(include_project_docs=False, include_file_tree=False),
        memory_store=store,
    ).build(
        task="continue",
    )

    frames = {frame.kind: frame for frame in context.frames()}
    assert frames["memory"].payload["scratchpad"]["project_goal"] == "keep raw"
    assert frames["handoff"].payload["text"] == "handoff text"
    assert "file_summaries" not in frames
    assert "<memory_anchor>" not in json.dumps([frame.payload for frame in context.frames()])


def test_parse_compaction_response_accepts_handoff_and_memories_json():
    payload = parse_compaction_response(
        json.dumps(
            {
                "handoff": "## 目标\n继续任务",
                "memories": [
                    {
                        "type": "procedural",
                        "content": "修改工具 schema 后需要同步 tests/test_tools.py。",
                        "confidence": 0.8,
                        "reason": "这是可复用的测试维护经验。",
                    },
                    {"type": "temporary", "content": "skip", "confidence": 0.8, "reason": "bad type"},
                ],
            },
            ensure_ascii=False,
        )
    )

    assert payload.handoff == "## 目标\n继续任务"
    assert payload.memories == [
        {
            "type": "procedural",
            "content": "修改工具 schema 后需要同步 tests/test_tools.py。",
            "confidence": 0.8,
            "reason": "这是可复用的测试维护经验。",
        }
    ]


def test_parse_compaction_response_falls_back_to_handoff_only():
    payload = parse_compaction_response("## 目标\n继续任务")

    assert payload.handoff == "## 目标\n继续任务"
    assert payload.memories == []


class CompactClient:
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        return {"message": {"role": "assistant", "content": "## 当前目标\n继续任务"}}


def test_context_has_no_side_effectful_compact_method():
    assert not hasattr(Context, "compact")


def test_clear_old_tool_results_truncates_long_tool_result_without_mutating_raw_history(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(
            include_project_docs=False,
            include_file_tree=False,
            max_input_tokens=100,
            compact_threshold_ratio=0.1,
            protected_recent_turns=2,
        ),
        task="task",
    )
    long_tool = {"role": "tool", "tool_call_id": "call-1", "content": "x" * 5000}
    context.add_message({"role": "user", "content": "new"})
    context.add_message(long_tool)

    context.replace_active_history(None, [{"role": "user", "content": "new"}, {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "x" * 4000 + "\n... [tool result truncated after compaction]",
    }])

    assert context.raw_messages()[1]["content"] == "x" * 5000
    active_tool = next(frame for frame in context.frames() if frame.role == "tool")
    assert active_tool.payload["content"].endswith("... [tool result truncated after compaction]")


def test_replace_active_history_inserts_compact_summary_as_assistant_message(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(include_project_docs=False, include_file_tree=False),
        task="task",
    )
    context.add_message({"role": "user", "content": "old"})

    context.replace_active_history(
        {
            "summary": "## 目标\n继续实现上下文压缩",
            "archive_log_path": "dialog/archive.jsonl",
            "instruction": "Use read_file if exact history is needed.",
        },
        [{"role": "user", "content": "recent"}],
    )

    history = context.history_frames()
    assert history[0].kind == "compact_summary"
    assert history[0].role == "assistant"
    assert history[0].payload["role"] == "assistant"
    assert history[0].payload["content"].startswith("[CONTEXT COMPACTION]")
    assert "dialog/archive.jsonl" in history[0].payload["content"]
    assert "## 目标\n继续实现上下文压缩" in history[0].payload["content"]
    assert history[1].payload == {"role": "user", "content": "recent"}
    assert context.raw_messages() == [{"role": "user", "content": "old"}]


def test_context_slice_old_history_protects_recent_without_archiving(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(
            include_project_docs=False,
            include_file_tree=False,
            max_input_tokens=100,
            compact_threshold_ratio=0.1,
            protected_recent_turns=1,
            protected_tool_results=0,
        ),
        task="task",
    )
    prior = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "x" * 5000},
    ]
    for message in prior:
        context.add_message(message)

    old_messages, remaining_messages = context.slice_old_history()

    assert context.raw_messages() == prior
    assert old_messages == prior[:2]
    assert remaining_messages[0]["content"] == "new"
    assert "旧 tool 输出已清理" in remaining_messages[2]["content"]


def test_context_slice_old_history_aligns_tool_result_with_parent_call(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(
            include_project_docs=False,
            include_file_tree=False,
            protected_recent_turns=1,
            protected_tool_results=10,
        ),
        task="task",
    )
    prior = [
        {"role": "user", "content": "old"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "read_file", "content": "result"},
        {"role": "user", "content": "new"},
    ]
    for message in prior:
        context.add_message(message)

    old_messages, remaining_messages = context.slice_old_history()

    assert old_messages == [{"role": "user", "content": "old"}]
    assert remaining_messages[0]["role"] == "assistant"
    assert remaining_messages[1]["role"] == "tool"
    assert remaining_messages[2] == {"role": "user", "content": "new"}
