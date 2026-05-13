import subprocess

from coding_agent.context import WorkspaceContext, WorkspaceContextOptions


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
