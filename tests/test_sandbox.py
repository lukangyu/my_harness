from pathlib import Path

import pytest

from coding_agent.sandbox import SandboxError, WorkspaceSandbox


def test_resolve_allows_project_file(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    project_file = tmp_path / "src" / "coding_agent" / "sandbox.py"
    project_file.parent.mkdir(parents=True)
    project_file.write_text("", encoding="utf-8")

    assert sandbox.resolve(Path("src") / "coding_agent" / "sandbox.py") == project_file.resolve()


def test_resolve_rejects_parent_traversal(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)

    with pytest.raises(SandboxError, match="outside workspace"):
        sandbox.resolve("../outside.txt")


def test_resolve_rejects_absolute_path_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    workspace.mkdir()
    outside.write_text("", encoding="utf-8")
    sandbox = WorkspaceSandbox(workspace)

    with pytest.raises(SandboxError, match="outside workspace"):
        sandbox.resolve(outside)


def test_relative_path_returns_posix_path(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    nested_file = tmp_path / "src" / "coding_agent" / "sandbox.py"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("", encoding="utf-8")

    assert sandbox.relative_path(nested_file) == "src/coding_agent/sandbox.py"
