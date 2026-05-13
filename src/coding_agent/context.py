from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DOC_NAMES = (
    "AGENTS.md",
    ".cursorrules",
    "README.md",
    "pyproject.toml",
    "package.json",
    "Makefile",
    "justfile",
    "pytest.ini",
    "tox.ini",
    "Cargo.toml",
    "go.mod",
)

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".coding-agent",
}


@dataclass(frozen=True)
class WorkspaceContextOptions:
    doc_max_chars: int = 1200
    tree_max_entries: int = 200
    include_project_docs: bool = True
    include_file_tree: bool = True
    include_git_status: bool = True
    include_recent_commits: bool = True


@dataclass(frozen=True)
class WorkspaceContext:
    cwd: Path
    repo_root: Path | None
    branch: str | None
    default_branch: str | None
    status: str
    recent_commits: list[str]
    project_docs: dict[str, str]
    file_tree: list[str]

    @classmethod
    def build(cls, cwd: Path, options: WorkspaceContextOptions) -> "WorkspaceContext":
        cwd = cwd.resolve()
        repo_root_text = _git(cwd, "rev-parse", "--show-toplevel")
        repo_root = Path(repo_root_text).resolve() if repo_root_text else None
        git_cwd = repo_root or cwd

        branch = _git(git_cwd, "branch", "--show-current") if repo_root else ""
        default_branch = _default_branch(git_cwd) if repo_root else None
        status = (
            _git(git_cwd, "status", "--short")
            if repo_root and options.include_git_status
            else ""
        )
        commits_text = (
            _git(git_cwd, "log", "--oneline", "-5")
            if repo_root and options.include_recent_commits
            else ""
        )

        return cls(
            cwd=cwd,
            repo_root=repo_root,
            branch=branch or None,
            default_branch=default_branch,
            status=status,
            recent_commits=commits_text.splitlines() if commits_text else [],
            project_docs=_project_docs(cwd, repo_root, options)
            if options.include_project_docs
            else {},
            file_tree=_file_tree(repo_root or cwd, options) if options.include_file_tree else [],
        )

    def fingerprint(self) -> str:
        return _hash_json(
            {
                "cwd": str(self.cwd),
                "repo_root": str(self.repo_root) if self.repo_root else None,
                "branch": self.branch,
                "default_branch": self.default_branch,
                "status": self.status,
                "recent_commits": self.recent_commits,
                "project_docs": self.project_docs,
                "file_tree": self.file_tree,
            }
        )


def _git(cwd: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=5,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _default_branch(cwd: Path) -> str | None:
    origin_head = _git(cwd, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    if origin_head.startswith("origin/"):
        return origin_head.removeprefix("origin/")
    for name in ("main", "master"):
        if _git(cwd, "rev-parse", "--verify", name):
            return name
    return None


def _project_docs(
    cwd: Path, repo_root: Path | None, options: WorkspaceContextOptions
) -> dict[str, str]:
    docs: dict[str, str] = {}
    root = repo_root or cwd
    for base in (repo_root, cwd):
        if base is None:
            continue
        for name in DOC_NAMES:
            path = base / name
            if not path.is_file():
                continue
            try:
                key = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                key = path.name
            if key in docs:
                continue
            try:
                docs[key] = _clip(
                    path.read_text(encoding="utf-8", errors="replace"),
                    options.doc_max_chars,
                )
            except OSError:
                continue
    return docs


def _file_tree(root: Path, options: WorkspaceContextOptions) -> list[str]:
    if options.tree_max_entries <= 0:
        return []

    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        relative_dir = current.relative_to(root)
        if _is_ignored(relative_dir):
            dirnames[:] = []
            continue
        dirnames[:] = sorted(name for name in dirnames if not _is_ignored(relative_dir / name))
        for filename in sorted(filenames):
            path = current / filename
            relative = path.relative_to(root)
            if _is_ignored(relative):
                continue
            files.append(relative.as_posix())
            if len(files) >= options.tree_max_entries:
                return files
    return files


def _is_ignored(relative: Path) -> bool:
    return any(part in IGNORED_DIRS or part.startswith(".pytest") for part in relative.parts)


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
