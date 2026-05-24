import json

from coding_agent.memory.store import MemoryStore


def test_memory_store_records_file_tool_facts(tmp_path):
    store = MemoryStore(tmp_path)

    store.record_tool_result(
        tool="write_file",
        arguments={"path": "notes.txt", "content": "hello"},
        result={"ok": True, "path": "notes.txt"},
    )
    store.record_tool_result(
        tool="read_file",
        arguments={"path": "notes.txt"},
        result={"ok": True, "path": "notes.txt", "content": "hello"},
    )

    scratchpad = store.load_scratchpad()
    assert scratchpad["modified_files"] == ["notes.txt"]
    assert scratchpad["read_files"] == ["notes.txt"]


def test_memory_store_uses_injected_memory_dir(tmp_path):
    memory_dir = tmp_path / ".coding-agent" / "conversations" / "c1" / "sessions" / "s1" / "memory"
    store = MemoryStore(tmp_path, memory_dir=memory_dir)

    store.write_handoff("handoff")
    store.save_scratchpad({"active_todos": ["continue"]})

    assert store.memory_dir == memory_dir
    assert (memory_dir / "handoff.md").read_text(encoding="utf-8").strip() == "handoff"
    assert json.loads((memory_dir / "scratchpad.json").read_text(encoding="utf-8"))["active_todos"] == ["continue"]
    assert not (tmp_path / ".coding-agent" / "memory").exists()


def test_memory_store_records_shell_failure_as_known_issue(tmp_path):
    store = MemoryStore(tmp_path)

    store.record_tool_result(
        tool="run_shell",
        arguments={"command": "pytest"},
        result={"ok": False, "command": "pytest", "exit_code": 1, "stderr": "failed"},
    )

    scratchpad = store.load_scratchpad()
    assert scratchpad["last_verified_commands"] == [
        {"command": "pytest", "ok": False, "exit_code": 1, "timed_out": None}
    ]
    assert scratchpad["known_issues"][0]["error"] == "failed"


def test_memory_anchor_renders_json_block(tmp_path):
    store = MemoryStore(tmp_path)
    store.save_scratchpad({"project_goal": "优化上下文", "modified_files": ["a.py"]})

    anchor = store.render_memory_anchor(max_chars=1000)

    assert anchor.startswith("<memory_anchor>")
    assert "优化上下文" in anchor
    assert json.loads(anchor.removeprefix("<memory_anchor>\n").removesuffix("\n</memory_anchor>"))["modified_files"] == ["a.py"]


def test_memory_store_does_not_create_file_summary_or_tool_index_cache(tmp_path):
    store = MemoryStore(tmp_path)

    store.record_tool_result(
        tool="write_file",
        arguments={"path": "module.py"},
        result={"ok": True, "path": "module.py"},
    )

    assert not (store.memory_dir / "file_summaries.json").exists()
    assert not (store.memory_dir / "tool_index.jsonl").exists()


def test_memory_store_archives_dialog_messages_as_jsonl(tmp_path):
    store = MemoryStore(tmp_path)

    path = store.archive_dialog_messages(
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "answer"},
        ]
    )

    assert path is not None
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"role": "user", "content": "old"}
    assert path.parent == store.dialog_dir


def test_memory_store_offloads_long_tool_result(tmp_path):
    store = MemoryStore(tmp_path)

    result = store.offload_tool_result(tool="read_file", content="x" * 5000, max_inline_chars=100)

    assert result["offloaded"] is True
    assert result["original_chars"] == 5000
    assert "完整 tool 输出已转存" in result["content"]
    assert len(result["content"]) < 300
    assert (tmp_path / result["path"]).read_text(encoding="utf-8") == "x" * 5000


def test_memory_store_keeps_short_tool_result_inline(tmp_path):
    store = MemoryStore(tmp_path)

    result = store.offload_tool_result(tool="read_file", content="short", max_inline_chars=100)

    assert result == {"content": "short", "offloaded": False}


def test_memory_store_appends_long_term_memories_to_conversation_raw_jsonl(tmp_path):
    conversation_memory_dir = tmp_path / ".coding-agent" / "conversations" / "c1" / "memory"
    store = MemoryStore(tmp_path, conversation_memory_dir=conversation_memory_dir)

    written = store.append_long_term_memories(
        [
            {
                "type": "procedural",
                "content": "修改工具 schema 后需要同步 tests/test_tools.py。",
                "reason": "本轮修改工具参数后测试需要更新。",
                "confidence": 0.8,
            }
        ],
        source="context_compaction",
        evidence=["sessions/s1/runs/r1/dialog/archive.jsonl"],
    )

    raw_files = list((conversation_memory_dir / "raw").glob("*.jsonl"))
    assert len(raw_files) == 1
    record = json.loads(raw_files[0].read_text(encoding="utf-8").strip())
    assert written[0]["id"] == record["id"]
    assert record["type"] == "procedural"
    assert record["source"] == "context_compaction"
    assert record["evidence"] == ["sessions/s1/runs/r1/dialog/archive.jsonl"]


def test_memory_store_filters_invalid_long_term_memories(tmp_path):
    store = MemoryStore(tmp_path, conversation_memory_dir=tmp_path / "conversation-memory")

    written = store.append_long_term_memories(
        [
            {"type": "temporary", "content": "skip", "reason": "bad type", "confidence": 0.9},
            {"type": "knowledge", "content": "", "reason": "empty", "confidence": 0.9},
            {"type": "personal", "content": "用户偏好中文说明。", "reason": "明确要求", "confidence": 0.1},
        ],
        source="context_compaction",
        evidence=[],
    )

    assert written == []
    assert not (tmp_path / "conversation-memory" / "raw").exists()
