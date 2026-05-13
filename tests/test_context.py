import subprocess

from coding_agent.context import (
    ContextEnvelope,
    PromptBuilder,
    StablePrefixManager,
    WorkspaceContext,
    WorkspaceContextOptions,
    WorkspacePrefixManager,
    WorkspacePrefixState,
    render_workspace_context,
)


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
