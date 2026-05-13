from __future__ import annotations

import hashlib
import html
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

PROMPT_VERSION = "context-v1"
STABLE_PREFIX_TEMPLATE = """<coding_agent_prefix version="context-v1">
<identity>
You are coding-agent, a local AI coding assistant.
</identity>

<operating_principles>
- Inspect relevant files before editing.
- Prefer small, focused changes.
- Verify changes with allowed commands when practical.
- Do not claim tests passed unless tool results show they passed.
</operating_principles>

<safety_contract>
- Work only through provided tools.
- File access is limited by the workspace sandbox.
- Shell commands are subject to command policy.
- If a tool is rejected or fails, report the reason and adapt.
</safety_contract>

<tool_contract>
- Use list_files/read_file/search_text to inspect.
- Use write_file only for intentional file writes.
- run_shell may be rejected by policy.
- apply_patch is currently unsupported.
</tool_contract>

<response_contract>
- Final answers should summarize changes, files touched, verification, and residual risk.
</response_contract>
</coding_agent_prefix>"""


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


@dataclass(frozen=True)
class StablePrefixState:
    text: str
    system_hash: str
    tool_signature: str
    rules_hash: str
    prompt_cache_key: str
    prompt_version: str


class StablePrefixManager:
    def __init__(self) -> None:
        self._state: StablePrefixState | None = None

    def get_or_build(self, tool_schemas: list[dict[str, Any]]) -> StablePrefixState:
        system_hash = _hash_text(STABLE_PREFIX_TEMPLATE)
        tool_signature = _hash_json(tool_schemas)
        rules_hash = _hash_text("")
        prompt_cache_key = _hash_text(
            PROMPT_VERSION + system_hash + tool_signature + rules_hash
        )
        if self._state and self._state.prompt_cache_key == prompt_cache_key:
            return self._state
        self._state = StablePrefixState(
            text=STABLE_PREFIX_TEMPLATE,
            system_hash=system_hash,
            tool_signature=tool_signature,
            rules_hash=rules_hash,
            prompt_cache_key=prompt_cache_key,
            prompt_version=PROMPT_VERSION,
        )
        return self._state


@dataclass(frozen=True)
class WorkspacePrefixState:
    text: str
    workspace_fingerprint: str
    cwd: str
    repo_root: str | None
    branch: str | None
    default_branch: str | None
    status_hash: str
    recent_commits_hash: str
    project_docs_hash: str
    file_tree_hash: str


@dataclass(frozen=True)
class ContextEnvelope:
    stable_prefix: StablePrefixState
    workspace_prefix: WorkspacePrefixState
    session_summary: str | None
    recent_messages: list[dict[str, Any]]
    current_task: str
    full_context_key: str


class PromptBuilder:
    def to_messages(self, envelope: ContextEnvelope, mode: str) -> list[dict[str, Any]]:
        messages = [
            {"role": "system", "content": envelope.stable_prefix.text},
            {"role": "user", "content": envelope.workspace_prefix.text},
        ]
        if envelope.session_summary:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"<session_summary>\n"
                        f"{envelope.session_summary}\n"
                        f"</session_summary>"
                    ),
                }
            )
        messages.extend(envelope.recent_messages)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"<current_task>\n"
                    f"mode: {mode}\n"
                    f"content:\n{envelope.current_task}\n"
                    f"</current_task>"
                ),
            }
        )
        return messages


class WorkspacePrefixManager:
    def __init__(self) -> None:
        self._state: WorkspacePrefixState | None = None

    def get_or_build(self, context: WorkspaceContext) -> WorkspacePrefixState:
        fingerprint = context.fingerprint()
        if self._state and self._state.workspace_fingerprint == fingerprint:
            return self._state
        text = render_workspace_context(context)
        self._state = WorkspacePrefixState(
            text=text,
            workspace_fingerprint=fingerprint,
            cwd=str(context.cwd),
            repo_root=str(context.repo_root) if context.repo_root else None,
            branch=context.branch,
            default_branch=context.default_branch,
            status_hash=_hash_text(context.status),
            recent_commits_hash=_hash_json(context.recent_commits),
            project_docs_hash=_hash_json(context.project_docs),
            file_tree_hash=_hash_json(context.file_tree),
        )
        return self._state


def render_workspace_context(context: WorkspaceContext) -> str:
    lines = [
        "<workspace_context>",
        f"cwd: {context.cwd}",
        f"repo_root: {context.repo_root if context.repo_root else ''}",
        f"is_git_repo: {str(context.repo_root is not None).lower()}",
        f"branch: {context.branch or ''}",
        f"default_branch: {context.default_branch or ''}",
        "",
        "git_status:",
    ]
    lines.extend([f"  {line}" for line in context.status.splitlines()] or ["  clean"])
    lines.append("")
    lines.append("recent_commits:")
    lines.extend([f"  {line}" for line in context.recent_commits] or ["  none"])
    lines.append("")
    lines.append("file_tree:")
    lines.extend([f"  {line}" for line in context.file_tree] or ["  none"])
    lines.append("")
    lines.append("project_docs:")
    if context.project_docs:
        for path, content in sorted(context.project_docs.items()):
            lines.append(f'  <doc path="{html.escape(path)}">')
            lines.extend(f"  {html.escape(line)}" for line in content.splitlines())
            lines.append("  </doc>")
    else:
        lines.append("  none")
    lines.append("</workspace_context>")
    return "\n".join(lines)


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


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
