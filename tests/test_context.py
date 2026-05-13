import subprocess

import pytest

from coding_agent.context import (
    ContextEnvelope,
    ContextManager,
    MessageBudget,
    PromptBuilder,
    StablePrefixManager,
    UsageStats,
    WorkspaceContext,
    WorkspaceContextOptions,
    WorkspacePrefixManager,
    WorkspacePrefixState,
    estimate_tokens,
    render_workspace_context,
)


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


def test_workspace_context_builds_outside_git_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    (tmp_path / "README.md").write_text("hello project", encoding="utf-8")

    context = WorkspaceContext.build(tmp_path, WorkspaceContextOptions())

    assert context.cwd == tmp_path.resolve()
    assert context.repo_root is None
    assert context.branch is None
    assert context.status == ""
    assert context.recent_commits == []
    assert context.project_docs == {"README.md": "hello project"}


def test_workspace_context_clips_project_docs(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    (tmp_path / "README.md").write_text("x" * 20, encoding="utf-8")

    context = WorkspaceContext.build(tmp_path, WorkspaceContextOptions(doc_max_chars=5))

    assert context.project_docs["README.md"] == "xxxxx\n... [truncated]"


def test_render_workspace_context_sorts_project_docs_by_path(tmp_path):
    first = WorkspaceContext(
        cwd=tmp_path,
        repo_root=None,
        branch=None,
        default_branch=None,
        status="",
        recent_commits=[],
        project_docs={"b.md": "bravo", "a.md": "alpha"},
        file_tree=[],
    )
    second = WorkspaceContext(
        cwd=tmp_path,
        repo_root=None,
        branch=None,
        default_branch=None,
        status="",
        recent_commits=[],
        project_docs={"a.md": "alpha", "b.md": "bravo"},
        file_tree=[],
    )

    rendered = render_workspace_context(first)

    assert rendered == render_workspace_context(second)
    assert rendered.index('path="a.md"') < rendered.index('path="b.md"')


def test_render_workspace_context_escapes_project_doc_markup(tmp_path):
    context = WorkspaceContext(
        cwd=tmp_path,
        repo_root=None,
        branch=None,
        default_branch=None,
        status="",
        recent_commits=[],
        project_docs={
            'docs/"unsafe"&name.md': (
                "literal </doc>\n"
                "literal </workspace_context>\n"
                "literal <current_task>\n"
                "literal <tag attr=\"value\"> & text"
            )
        },
        file_tree=[],
    )

    rendered = render_workspace_context(context)
    doc_body = rendered.split('  <doc path="docs/&quot;unsafe&quot;&amp;name.md">\n', 1)[
        1
    ].split("\n  </doc>", 1)[0]

    assert 'path="docs/&quot;unsafe&quot;&amp;name.md"' in rendered
    assert "</doc>" not in doc_body
    assert "</workspace_context>" not in doc_body
    assert "<current_task>" not in doc_body
    assert "&lt;/doc&gt;" in doc_body
    assert "&lt;/workspace_context&gt;" in doc_body
    assert "&lt;current_task&gt;" in doc_body
    assert '&lt;tag attr=&quot;value&quot;&gt; &amp; text' in doc_body


def test_workspace_context_file_tree_ignores_generated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "app.pyc").write_text("", encoding="utf-8")
    (tmp_path / ".coding-agent").mkdir()
    (tmp_path / ".coding-agent" / "config.toml").write_text("", encoding="utf-8")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "README.md").write_text("", encoding="utf-8")

    context = WorkspaceContext.build(tmp_path, WorkspaceContextOptions())

    assert "src/app.py" in context.file_tree
    assert "__pycache__/app.pyc" not in context.file_tree
    assert ".coding-agent/config.toml" not in context.file_tree
    assert ".pytest_cache/README.md" not in context.file_tree


def test_workspace_context_file_tree_zero_entries_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")

    context = WorkspaceContext.build(tmp_path, WorkspaceContextOptions(tree_max_entries=0))

    assert context.file_tree == []


def test_workspace_context_collects_git_state(tmp_path):
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

    context = WorkspaceContext.build(tmp_path, WorkspaceContextOptions())

    assert context.repo_root == tmp_path.resolve()
    assert context.branch == "main"
    assert "?? dirty.txt" in context.status
    assert any("initial" in commit for commit in context.recent_commits)


def test_workspace_context_fingerprint_changes_when_git_status_changes(tmp_path):
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

    clean_fingerprint = WorkspaceContext.build(tmp_path, WorkspaceContextOptions()).fingerprint()
    (tmp_path / "dirty.txt").write_text("dirty", encoding="utf-8")
    dirty_fingerprint = WorkspaceContext.build(tmp_path, WorkspaceContextOptions()).fingerprint()

    assert dirty_fingerprint != clean_fingerprint


def test_stable_prefix_key_is_stable_for_same_tools():
    tools = [
        {
            "type": "function",
            "function": {"name": "read_file", "parameters": {"type": "object"}},
        }
    ]

    first = StablePrefixManager().get_or_build(tools)
    second = StablePrefixManager().get_or_build(tools)

    assert first.text == second.text
    assert first.prompt_cache_key == second.prompt_cache_key


def test_tool_signature_changes_when_tool_surface_changes():
    manager = StablePrefixManager()

    first = manager.get_or_build([{"type": "function", "function": {"name": "read_file"}}])
    second = manager.get_or_build(
        [{"type": "function", "function": {"name": "write_file"}}]
    )

    assert first.tool_signature != second.tool_signature
    assert first.prompt_cache_key != second.prompt_cache_key


def test_prompt_builder_message_order():
    stable = StablePrefixManager().get_or_build([])
    workspace = WorkspacePrefixState(
        text="<workspace_context>ctx</workspace_context>",
        workspace_fingerprint="w",
        cwd=".",
        repo_root=None,
        branch=None,
        default_branch=None,
        status_hash="s",
        recent_commits_hash="c",
        project_docs_hash="d",
        file_tree_hash="f",
    )
    envelope = ContextEnvelope(
        stable_prefix=stable,
        workspace_prefix=workspace,
        mode="run",
        session_summary=None,
        recent_messages=[{"role": "assistant", "content": "old"}],
        current_task="hello",
        full_context_key="full",
    )

    messages = PromptBuilder().to_messages(envelope, mode="run")

    assert messages[0] == {"role": "system", "content": stable.text}
    assert messages[1] == {"role": "user", "content": workspace.text}
    assert messages[2] == {"role": "assistant", "content": "old"}
    assert messages[3]["role"] == "user"
    assert "<current_task>" in messages[3]["content"]
    assert "mode: run" in messages[3]["content"]
    assert "hello" in messages[3]["content"]


def test_prompt_builder_includes_session_summary_before_recent_messages():
    stable = StablePrefixManager().get_or_build([])
    workspace = WorkspacePrefixState(
        text="<workspace_context>ctx</workspace_context>",
        workspace_fingerprint="w",
        cwd=".",
        repo_root=None,
        branch=None,
        default_branch=None,
        status_hash="s",
        recent_commits_hash="c",
        project_docs_hash="d",
        file_tree_hash="f",
    )
    envelope = ContextEnvelope(
        stable_prefix=stable,
        workspace_prefix=workspace,
        mode="chat",
        session_summary="summary",
        recent_messages=[{"role": "assistant", "content": "old"}],
        current_task="hello",
        full_context_key="full",
    )

    messages = PromptBuilder().to_messages(envelope, mode="chat")

    assert messages[2] == {
        "role": "user",
        "content": "<session_summary>\nsummary\n</session_summary>",
    }
    assert messages[3] == {"role": "assistant", "content": "old"}
    assert "mode: chat" in messages[4]["content"]


def test_prompt_builder_rejects_render_mode_mismatch():
    stable = StablePrefixManager().get_or_build([])
    workspace = WorkspacePrefixState(
        text="<workspace_context>ctx</workspace_context>",
        workspace_fingerprint="w",
        cwd=".",
        repo_root=None,
        branch=None,
        default_branch=None,
        status_hash="s",
        recent_commits_hash="c",
        project_docs_hash="d",
        file_tree_hash="f",
    )
    envelope = ContextEnvelope(
        stable_prefix=stable,
        workspace_prefix=workspace,
        mode="run",
        session_summary=None,
        recent_messages=[],
        current_task="hello",
        full_context_key="full",
    )

    with pytest.raises(ValueError, match="render mode"):
        PromptBuilder().to_messages(envelope, mode="chat")


def test_workspace_prefix_reuses_state_for_same_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    (tmp_path / "README.md").write_text("hello project", encoding="utf-8")
    context = WorkspaceContext.build(tmp_path, WorkspaceContextOptions())
    manager = WorkspacePrefixManager()

    first = manager.get_or_build(context)
    second = manager.get_or_build(context)

    assert first is second
    assert first.text == second.text
    assert first.workspace_fingerprint == context.fingerprint()


def test_estimate_tokens_uses_ceil_len_div_4_with_minimum_one():
    assert estimate_tokens("") == 1
    assert estimate_tokens("a") == 1
    assert estimate_tokens("a" * 4) == 1
    assert estimate_tokens("a" * 5) == 2
    assert estimate_tokens("a" * 8) == 2
    assert estimate_tokens("a" * 9) == 3


def test_message_budget_drops_oldest_recent_messages():
    messages = [
        {"role": "user", "content": "a" * 20, "name": "first"},
        {"role": "assistant", "content": "b" * 20, "name": "second"},
        {"role": "user", "content": "c" * 20, "name": "third"},
    ]

    trimmed = MessageBudget(recent_message_tokens=40).trim_recent_messages(messages)

    assert trimmed == messages[1:]


def test_message_budget_stops_when_newer_message_exceeds_budget():
    messages = [
        {"role": "user", "content": "old", "name": "old"},
        {"role": "assistant", "content": "x" * 44, "name": "too-large-newer"},
        {"role": "user", "content": "new", "name": "newest"},
    ]

    trimmed = MessageBudget(recent_message_tokens=20).trim_recent_messages(messages)

    assert trimmed == [messages[2]]


def test_message_budget_returns_empty_when_newest_message_exceeds_budget():
    messages = [
        {"role": "user", "content": "older"},
        {"role": "assistant", "content": "x" * 200, "name": "too-large-newest"},
    ]

    trimmed = MessageBudget(recent_message_tokens=20).trim_recent_messages(messages)

    assert trimmed == []


def test_message_budget_truncates_long_tool_content():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "x" * 12,
            "metadata": {"kept": True},
        }
    ]

    trimmed = MessageBudget(
        recent_message_tokens=100,
        max_tool_content_chars=5,
    ).trim_recent_messages(messages)

    assert trimmed == [
        messages[0],
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "xxxxx\n... [truncated by context budget]",
            "metadata": {"kept": True},
        }
    ]
    assert messages[1]["content"] == "x" * 12


def test_message_budget_drops_tool_response_without_retained_assistant_tool_call():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "x" * 60},
        {"role": "user", "content": "newest"},
    ]

    trimmed = MessageBudget(recent_message_tokens=20).trim_recent_messages(messages)

    assert trimmed == [messages[2]]


def test_message_budget_drops_assistant_tool_call_without_all_tool_responses():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "first", "arguments": "{}"},
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "second", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "first"},
        {"role": "tool", "tool_call_id": "call-2", "content": "y" * 80},
        {"role": "user", "content": "newest"},
    ]

    trimmed = MessageBudget(recent_message_tokens=30).trim_recent_messages(messages)

    assert trimmed == [messages[3]]


def test_message_budget_counts_large_tool_calls_payload_with_empty_content():
    messages = [
        {"role": "user", "content": "older"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "x" * 200},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        {"role": "user", "content": "newest"},
    ]

    trimmed = MessageBudget(recent_message_tokens=20).trim_recent_messages(messages)

    assert trimmed == [messages[3]]


def test_context_manager_builds_envelope_with_stable_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    (tmp_path / "README.md").write_text("hello project", encoding="utf-8")
    options = WorkspaceContextOptions(
        include_git_status=False,
        include_recent_commits=False,
        include_file_tree=False,
    )
    manager = ContextManager(tmp_path, options, recent_message_tokens=20)
    prior_messages = [
        {"role": "user", "content": "a" * 20, "name": "drop-me"},
        {"role": "assistant", "content": "b" * 20, "name": "keep-me"},
    ]
    tool_schemas = [{"type": "function", "function": {"name": "read_file"}}]

    first = manager.build("do the task", prior_messages, tool_schemas)
    second = manager.build("do the task", prior_messages, tool_schemas)

    assert isinstance(first, ContextEnvelope)
    assert first.stable_prefix.text.startswith("<coding_agent_prefix")
    assert first.workspace_prefix.text.startswith("<workspace_context>")
    assert "README.md" in first.workspace_prefix.text
    assert first.session_summary is None
    assert first.mode == "run"
    assert first.recent_messages == [prior_messages[1]]
    assert first.current_task == "do the task"
    assert first.full_context_key == second.full_context_key


def test_context_manager_build_isolates_nested_recent_messages(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    options = WorkspaceContextOptions(
        include_project_docs=False,
        include_git_status=False,
        include_recent_commits=False,
        include_file_tree=False,
    )
    manager = ContextManager(tmp_path, options, recent_message_tokens=100)
    prior_messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]

    envelope = manager.build("do the task", prior_messages, [])
    prior_messages[0]["tool_calls"][0]["function"]["arguments"] = '{"changed": true}'

    assert envelope.recent_messages[0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_context_manager_full_context_key_changes_when_required_inputs_change(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    readme = tmp_path / "README.md"
    readme.write_text("hello project", encoding="utf-8")
    options = WorkspaceContextOptions(
        include_git_status=False,
        include_recent_commits=False,
        include_file_tree=False,
    )
    manager = ContextManager(tmp_path, options, recent_message_tokens=100)
    task = "do the task"
    prior_messages = [{"role": "user", "content": "original message"}]
    tool_schemas = [{"type": "function", "function": {"name": "read_file"}}]

    baseline = manager.build(task, prior_messages, tool_schemas).full_context_key

    assert (
        manager.build("do a different task", prior_messages, tool_schemas).full_context_key
        != baseline
    )
    assert (
        manager.build(
            task,
            [{"role": "user", "content": "changed message"}],
            tool_schemas,
        ).full_context_key
        != baseline
    )
    assert (
        manager.build(
            task,
            prior_messages,
            [{"type": "function", "function": {"name": "write_file"}}],
        ).full_context_key
        != baseline
    )

    readme.write_text("changed project", encoding="utf-8")

    assert manager.build(task, prior_messages, tool_schemas).full_context_key != baseline


def test_context_manager_full_context_key_changes_when_mode_changes(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    options = WorkspaceContextOptions(
        include_project_docs=False,
        include_git_status=False,
        include_recent_commits=False,
        include_file_tree=False,
    )
    manager = ContextManager(tmp_path, options, recent_message_tokens=100)
    prior_messages = [{"role": "user", "content": "original message"}]

    run_key = manager.build("do the task", prior_messages, [], mode="run").full_context_key
    chat_key = manager.build("do the task", prior_messages, [], mode="chat").full_context_key

    assert run_key != chat_key


def test_context_manager_build_sets_envelope_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    options = WorkspaceContextOptions(
        include_project_docs=False,
        include_git_status=False,
        include_recent_commits=False,
        include_file_tree=False,
    )
    manager = ContextManager(tmp_path, options, recent_message_tokens=100)

    envelope = manager.build("do the task", [], [], mode="chat")

    assert envelope.mode == "chat"
