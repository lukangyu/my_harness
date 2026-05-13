# Context Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cache-friendly prompt context management with stable prefix state, workspace baseline injection, approximate-token message budgeting, basic session recovery, and usage/cache metadata parsing.

**Architecture:** Add a focused `coding_agent.context` module for context state, workspace sampling, prompt building, and budgeting. Keep `AgentLoop` responsible for tool-call execution, but delegate all request message construction to the context manager.

**Tech Stack:** Python 3.11+, stdlib dataclasses/hashlib/subprocess/pathlib/json, existing Typer/Rich/httpx/pytest stack.

---

## File Structure

- Create `src/coding_agent/context.py`: stable prefix, workspace context, workspace prefix, prompt builder, budget trimming, usage stats.
- Modify `src/coding_agent/config.py`: add optional `[context]` config with defaults.
- Modify `src/coding_agent/agent.py`: use `ContextManager` for message construction and expose usage stats in `AgentResult`.
- Modify `src/coding_agent/llm.py`: parse usage metadata into `UsageStats`.
- Modify `src/coding_agent/session.py`: add load/latest helpers.
- Modify `src/coding_agent/cli.py`: pass mode into agent, support chat resume flags, display cache stats when available.
- Add `tests/test_context.py`: deterministic unit coverage for the new context module.
- Update existing tests in `tests/test_config.py`, `tests/test_agent.py`, `tests/test_cli_run.py`.
- Update `README.md`: document context management behavior and config.

---

### Task 1: Context Config Defaults

**Files:**
- Modify: `src/coding_agent/config.py`
- Modify: `src/coding_agent/cli.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Add to `tests/test_config.py`:

```python
def test_load_config_uses_context_defaults_when_section_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_dir = tmp_path / ".coding-agent"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        """
[model]
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-4.1"

[agent]
max_steps = 20
stream = true

[workspace]
root = "."

[commands]
allow = []
deny = []
""",
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config.context.max_input_tokens == 24000
    assert config.context.reserved_output_tokens == 4000
    assert config.context.recent_message_tokens == 12000
    assert config.context.project_context_tokens == 4000
    assert config.context.doc_max_chars == 1200
    assert config.context.tree_max_entries == 200
    assert config.context.include_project_docs is True
    assert config.context.include_file_tree is True
    assert config.context.include_git_status is True
    assert config.context.include_recent_commits is True
    assert config.context.restore_last_session is False
    assert config.context.show_cache_stats is True


def test_load_config_reads_context_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_dir = tmp_path / ".coding-agent"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        """
[model]
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-4.1"

[agent]
max_steps = 20
stream = true

[workspace]
root = "."

[commands]
allow = []
deny = []

[context]
max_input_tokens = 1000
reserved_output_tokens = 100
recent_message_tokens = 500
project_context_tokens = 200
doc_max_chars = 300
tree_max_entries = 20
include_project_docs = false
include_file_tree = false
include_git_status = false
include_recent_commits = false
restore_last_session = true
show_cache_stats = false
""",
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config.context.max_input_tokens == 1000
    assert config.context.reserved_output_tokens == 100
    assert config.context.recent_message_tokens == 500
    assert config.context.project_context_tokens == 200
    assert config.context.doc_max_chars == 300
    assert config.context.tree_max_entries == 20
    assert config.context.include_project_docs is False
    assert config.context.include_file_tree is False
    assert config.context.include_git_status is False
    assert config.context.include_recent_commits is False
    assert config.context.restore_last_session is True
    assert config.context.show_cache_stats is False
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
C:\ProgramData\anaconda3\python.exe -m pytest tests/test_config.py -q --basetemp=.pytest-context-config-red
```

Expected: FAIL because `AppConfig` has no `context` field.

- [ ] **Step 3: Implement config dataclass and defaults**

Add to `src/coding_agent/config.py`:

```python
@dataclass(frozen=True)
class ContextConfig:
    max_input_tokens: int = 24000
    reserved_output_tokens: int = 4000
    recent_message_tokens: int = 12000
    project_context_tokens: int = 4000
    doc_max_chars: int = 1200
    tree_max_entries: int = 200
    include_project_docs: bool = True
    include_file_tree: bool = True
    include_git_status: bool = True
    include_recent_commits: bool = True
    restore_last_session: bool = False
    show_cache_stats: bool = True
```

Add `context: ContextConfig` to `AppConfig`.

Add helper:

```python
def _optional_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration key {key} must be a table")
    return value


def _optional_int(data: dict[str, Any], key: str, default: int, display_key: str) -> int:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"Configuration key {display_key} must be an integer")
    return value


def _optional_bool(data: dict[str, Any], key: str, default: bool, display_key: str) -> bool:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise ConfigError(f"Configuration key {display_key} must be a boolean")
    return value
```

In `load_config`, read:

```python
context_data = _optional_table(data, "context")
context_defaults = ContextConfig()
context = ContextConfig(
    max_input_tokens=_optional_int(context_data, "max_input_tokens", context_defaults.max_input_tokens, "context.max_input_tokens"),
    reserved_output_tokens=_optional_int(context_data, "reserved_output_tokens", context_defaults.reserved_output_tokens, "context.reserved_output_tokens"),
    recent_message_tokens=_optional_int(context_data, "recent_message_tokens", context_defaults.recent_message_tokens, "context.recent_message_tokens"),
    project_context_tokens=_optional_int(context_data, "project_context_tokens", context_defaults.project_context_tokens, "context.project_context_tokens"),
    doc_max_chars=_optional_int(context_data, "doc_max_chars", context_defaults.doc_max_chars, "context.doc_max_chars"),
    tree_max_entries=_optional_int(context_data, "tree_max_entries", context_defaults.tree_max_entries, "context.tree_max_entries"),
    include_project_docs=_optional_bool(context_data, "include_project_docs", context_defaults.include_project_docs, "context.include_project_docs"),
    include_file_tree=_optional_bool(context_data, "include_file_tree", context_defaults.include_file_tree, "context.include_file_tree"),
    include_git_status=_optional_bool(context_data, "include_git_status", context_defaults.include_git_status, "context.include_git_status"),
    include_recent_commits=_optional_bool(context_data, "include_recent_commits", context_defaults.include_recent_commits, "context.include_recent_commits"),
    restore_last_session=_optional_bool(context_data, "restore_last_session", context_defaults.restore_last_session, "context.restore_last_session"),
    show_cache_stats=_optional_bool(context_data, "show_cache_stats", context_defaults.show_cache_stats, "context.show_cache_stats"),
)
```

Include `context=context` in the returned `AppConfig`.

- [ ] **Step 4: Update DEFAULT_CONFIG**

Append this section to `DEFAULT_CONFIG` in `src/coding_agent/cli.py`:

```toml
[context]
max_input_tokens = 24000
reserved_output_tokens = 4000
recent_message_tokens = 12000
project_context_tokens = 4000
doc_max_chars = 1200
tree_max_entries = 200
include_project_docs = true
include_file_tree = true
include_git_status = true
include_recent_commits = true
restore_last_session = false
show_cache_stats = true
```

- [ ] **Step 5: Run tests**

Run:

```bash
C:\ProgramData\anaconda3\python.exe -m pytest tests/test_config.py tests/test_cli_init.py -q --basetemp=.pytest-context-config-green
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/coding_agent/config.py src/coding_agent/cli.py tests/test_config.py
git commit -m "feat: add context configuration"
```

---

### Task 2: Workspace Context and Fingerprints

**Files:**
- Create: `src/coding_agent/context.py`
- Test: `tests/test_context.py`

- [ ] **Step 1: Write failing workspace context tests**

Create `tests/test_context.py` with:

```python
from pathlib import Path

from coding_agent.context import WorkspaceContext, WorkspaceContextOptions


def test_workspace_context_builds_outside_git_repo(tmp_path):
    (tmp_path / "README.md").write_text("hello project", encoding="utf-8")

    context = WorkspaceContext.build(tmp_path, WorkspaceContextOptions())

    assert context.cwd == tmp_path.resolve()
    assert context.repo_root is None
    assert context.branch is None
    assert context.status == ""
    assert context.recent_commits == []
    assert context.project_docs == {"README.md": "hello project"}


def test_workspace_context_clips_project_docs(tmp_path):
    (tmp_path / "README.md").write_text("x" * 20, encoding="utf-8")

    context = WorkspaceContext.build(tmp_path, WorkspaceContextOptions(doc_max_chars=5))

    assert context.project_docs["README.md"] == "xxxxx\n... [truncated]"


def test_workspace_context_file_tree_ignores_generated_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "app.pyc").write_text("", encoding="utf-8")
    (tmp_path / ".coding-agent").mkdir()
    (tmp_path / ".coding-agent" / "config.toml").write_text("", encoding="utf-8")

    context = WorkspaceContext.build(tmp_path, WorkspaceContextOptions())

    assert "src/app.py" in context.file_tree
    assert "__pycache__/app.pyc" not in context.file_tree
    assert ".coding-agent/config.toml" not in context.file_tree
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
C:\ProgramData\anaconda3\python.exe -m pytest tests/test_context.py -q --basetemp=.pytest-workspace-context-red
```

Expected: FAIL because `coding_agent.context` does not exist.

- [ ] **Step 3: Implement workspace context**

Create `src/coding_agent/context.py` with:

```python
from __future__ import annotations

import hashlib
import json
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
        git_base = repo_root or cwd
        branch = _git(git_base, "branch", "--show-current") if repo_root else None
        default_branch = _default_branch(git_base) if repo_root else None
        status = _git(git_base, "status", "--short") if repo_root and options.include_git_status else ""
        commits_text = _git(git_base, "log", "--oneline", "-5") if repo_root and options.include_recent_commits else ""
        recent_commits = commits_text.splitlines() if commits_text else []
        project_docs = _project_docs(cwd, repo_root, options) if options.include_project_docs else {}
        file_tree = _file_tree(repo_root or cwd, options) if options.include_file_tree else []
        return cls(cwd, repo_root, branch or None, default_branch, status, recent_commits, project_docs, file_tree)

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
```

Append helpers:

```python
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


def _project_docs(cwd: Path, repo_root: Path | None, options: WorkspaceContextOptions) -> dict[str, str]:
    docs: dict[str, str] = {}
    bases = [base for base in (repo_root, cwd) if base is not None]
    root = repo_root or cwd
    for base in bases:
        for name in DOC_NAMES:
            path = base / name
            if not path.is_file():
                continue
            key = path.resolve().relative_to(root.resolve()).as_posix() if repo_root else path.name
            if key in docs:
                continue
            try:
                docs[key] = _clip(path.read_text(encoding="utf-8", errors="replace"), options.doc_max_chars)
            except OSError:
                continue
    return docs


def _file_tree(root: Path, options: WorkspaceContextOptions) -> list[str]:
    files: list[str] = []
    for path in sorted(root.rglob("*")):
        if len(files) >= options.tree_max_entries:
            break
        if any(part in IGNORED_DIRS or part.startswith(".pytest") for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            files.append(path.relative_to(root).as_posix())
    if len(files) >= options.tree_max_entries:
        files.append(f"... [truncated: showing first {options.tree_max_entries} entries]")
    return files


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run tests**

Run:

```bash
C:\ProgramData\anaconda3\python.exe -m pytest tests/test_context.py -q --basetemp=.pytest-workspace-context-green
```

Expected: PASS.

- [ ] **Step 5: Add git-specific tests**

Add:

```python
import subprocess


def test_workspace_context_collects_git_state(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "dirty.txt").write_text("dirty", encoding="utf-8")

    context = WorkspaceContext.build(tmp_path, WorkspaceContextOptions())

    assert context.repo_root == tmp_path.resolve()
    assert context.branch in {"main", "master"}
    assert "?? dirty.txt" in context.status
    assert any("initial" in commit for commit in context.recent_commits)
```

Run:

```bash
C:\ProgramData\anaconda3\python.exe -m pytest tests/test_context.py -q --basetemp=.pytest-workspace-context-git
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/coding_agent/context.py tests/test_context.py
git commit -m "feat: add workspace context baseline"
```

---

### Task 3: Prefix State and Prompt Builder

**Files:**
- Modify: `src/coding_agent/context.py`
- Test: `tests/test_context.py`

- [ ] **Step 1: Add failing tests for prefix state and prompt order**

Append to `tests/test_context.py`:

```python
from coding_agent.context import (
    ContextEnvelope,
    PromptBuilder,
    StablePrefixManager,
    WorkspacePrefixState,
)


def test_stable_prefix_key_is_stable_for_same_tools():
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]

    first = StablePrefixManager().get_or_build(tools)
    second = StablePrefixManager().get_or_build(tools)

    assert first.text == second.text
    assert first.prompt_cache_key == second.prompt_cache_key


def test_tool_signature_changes_when_tool_surface_changes():
    manager = StablePrefixManager()

    first = manager.get_or_build([{"type": "function", "function": {"name": "read_file"}}])
    second = manager.get_or_build([{"type": "function", "function": {"name": "write_file"}}])

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
    assert "hello" in messages[3]["content"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
C:\ProgramData\anaconda3\python.exe -m pytest tests/test_context.py -q --basetemp=.pytest-prefix-red
```

Expected: FAIL because prefix classes are missing.

- [ ] **Step 3: Implement prefix classes**

Append to `src/coding_agent/context.py`:

```python
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
        prompt_cache_key = _hash_text(PROMPT_VERSION + system_hash + tool_signature + rules_hash)
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
```

Add:

```python
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
            messages.append({"role": "user", "content": f"<session_summary>\n{envelope.session_summary}\n</session_summary>"})
        messages.extend(envelope.recent_messages)
        messages.append({"role": "user", "content": f"<current_task>\nmode: {mode}\ncontent:\n{envelope.current_task}\n</current_task>"})
        return messages


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Implement workspace prefix rendering**

Add:

```python
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
        for path, content in context.project_docs.items():
            lines.append(f'  <doc path="{path}">')
            lines.extend(f"  {line}" for line in content.splitlines())
            lines.append("  </doc>")
    else:
        lines.append("  none")
    lines.append("</workspace_context>")
    return "\n".join(lines)
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
C:\ProgramData\anaconda3\python.exe -m pytest tests/test_context.py -q --basetemp=.pytest-prefix-green
```

Expected: PASS.

Commit:

```bash
git add src/coding_agent/context.py tests/test_context.py
git commit -m "feat: add prompt prefix states"
```

---

### Task 4: Message Budget and Context Manager

**Files:**
- Modify: `src/coding_agent/context.py`
- Test: `tests/test_context.py`

- [ ] **Step 1: Add failing budget tests**

Append:

```python
from coding_agent.context import ContextManager, MessageBudget


def test_message_budget_drops_oldest_recent_messages():
    messages = [
        {"role": "user", "content": "old" * 100},
        {"role": "assistant", "content": "new"},
    ]

    trimmed = MessageBudget(recent_message_tokens=2).trim_recent_messages(messages)

    assert trimmed == [{"role": "assistant", "content": "new"}]


def test_message_budget_truncates_long_tool_content():
    messages = [{"role": "tool", "tool_call_id": "1", "name": "read_file", "content": "x" * 1000}]

    trimmed = MessageBudget(recent_message_tokens=20, max_tool_content_chars=40).trim_recent_messages(messages)

    assert trimmed[0]["role"] == "tool"
    assert trimmed[0]["tool_call_id"] == "1"
    assert trimmed[0]["name"] == "read_file"
    assert "[truncated by context budget]" in trimmed[0]["content"]
```

- [ ] **Step 2: Implement MessageBudget**

Add:

```python
class MessageBudget:
    def __init__(self, recent_message_tokens: int, max_tool_content_chars: int = 4000) -> None:
        self.recent_message_tokens = recent_message_tokens
        self.max_tool_content_chars = max_tool_content_chars

    def trim_recent_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared = [self._truncate_tool_message(message) for message in messages]
        kept: list[dict[str, Any]] = []
        total = 0
        for message in reversed(prepared):
            cost = estimate_tokens(message.get("content", ""))
            if kept and total + cost > self.recent_message_tokens:
                break
            if total + cost <= self.recent_message_tokens:
                kept.append(message)
                total += cost
        return list(reversed(kept))

    def _truncate_tool_message(self, message: dict[str, Any]) -> dict[str, Any]:
        if message.get("role") != "tool":
            return dict(message)
        content = str(message.get("content", ""))
        if len(content) <= self.max_tool_content_chars:
            return dict(message)
        updated = dict(message)
        updated["content"] = content[: self.max_tool_content_chars] + "\n... [truncated by context budget]"
        return updated


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)
```

- [ ] **Step 3: Add ContextManager**

Add:

```python
class ContextManager:
    def __init__(
        self,
        cwd: Path,
        options: WorkspaceContextOptions,
        recent_message_tokens: int,
    ) -> None:
        self.cwd = cwd
        self.options = options
        self.stable_prefix = StablePrefixManager()
        self.workspace_prefix = WorkspacePrefixManager()
        self.budget = MessageBudget(recent_message_tokens=recent_message_tokens)

    def build(
        self,
        task: str,
        prior_messages: list[dict[str, Any]] | None,
        tool_schemas: list[dict[str, Any]],
    ) -> ContextEnvelope:
        stable = self.stable_prefix.get_or_build(tool_schemas)
        workspace_context = WorkspaceContext.build(self.cwd, self.options)
        workspace = self.workspace_prefix.get_or_build(workspace_context)
        recent = self.budget.trim_recent_messages(list(prior_messages or []))
        full_key = _hash_text(
            stable.prompt_cache_key
            + workspace.workspace_fingerprint
            + _hash_json(recent)
            + _hash_text(task)
        )
        return ContextEnvelope(stable, workspace, None, recent, task, full_key)
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
C:\ProgramData\anaconda3\python.exe -m pytest tests/test_context.py -q --basetemp=.pytest-context-manager
```

Expected: PASS.

Commit:

```bash
git add src/coding_agent/context.py tests/test_context.py
git commit -m "feat: add context manager budget"
```

---

### Task 5: AgentLoop Integration

**Files:**
- Modify: `src/coding_agent/agent.py`
- Modify: `src/coding_agent/cli.py`
- Test: `tests/test_agent.py`
- Test: `tests/test_cli_run.py`

- [ ] **Step 1: Add failing agent tests for context-managed messages**

Update `tests/test_agent.py` expectations:

```python
def test_agent_sends_stable_prefix_and_workspace_context(tmp_path):
    client = FakeClient([{"message": {"role": "assistant", "content": "done"}}])
    agent = AgentLoop(client, make_tools(tmp_path), max_steps=1, cwd=tmp_path)

    result = agent.run("inspect")

    assert result.final_answer == "done"
    sent = client.calls[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "<coding_agent_prefix" in sent[0]["content"]
    assert sent[1]["role"] == "user"
    assert "<workspace_context>" in sent[1]["content"]
    assert sent[-1]["content"].count("<current_task>") == 1
```

- [ ] **Step 2: Modify AgentLoop constructor and run**

In `src/coding_agent/agent.py`:

- Import `ContextManager`, `PromptBuilder`, `WorkspaceContextOptions`.
- Remove `SYSTEM_PROMPT`.
- Update `AgentLoop.__init__`:

```python
def __init__(
    self,
    client: ChatClient,
    tools: ToolRegistry,
    max_steps: int,
    cwd: Path,
    context_options: WorkspaceContextOptions | None = None,
    recent_message_tokens: int = 12000,
) -> None:
    self.client = client
    self.tools = tools
    self.max_steps = max_steps
    self.context = ContextManager(
        cwd=cwd,
        options=context_options or WorkspaceContextOptions(),
        recent_message_tokens=recent_message_tokens,
    )
    self.prompt_builder = PromptBuilder()
```

Update `run`:

```python
def run(self, task: str, prior_messages: list[dict[str, Any]] | None = None, mode: str = "run") -> AgentResult:
    envelope = self.context.build(task, prior_messages, self.tools.schemas())
    messages = self.prompt_builder.to_messages(envelope, mode=mode)
    ...
```

Keep the existing tool-call loop unchanged after initial `messages` creation.

- [ ] **Step 3: Update CLI construction**

In `_run_task`, pass:

```python
context_options = WorkspaceContextOptions(
    doc_max_chars=config.context.doc_max_chars,
    tree_max_entries=config.context.tree_max_entries,
    include_project_docs=config.context.include_project_docs,
    include_file_tree=config.context.include_file_tree,
    include_git_status=config.context.include_git_status,
    include_recent_commits=config.context.include_recent_commits,
)
agent = AgentLoop(
    client=client,
    tools=tools,
    max_steps=config.agent.max_steps,
    cwd=Path.cwd(),
    context_options=context_options,
    recent_message_tokens=config.context.recent_message_tokens,
)
result = agent.run(task, prior_messages=prior_messages, mode=mode)
```

Change `_run_task` signature:

```python
def _run_task(task: str, prior_messages: list[dict[str, Any]] | None, mode: str) -> tuple[AgentResult, Path]:
```

Call with `mode="run"` or `mode="chat"`.

- [ ] **Step 4: Update tests**

Update existing `AgentLoop(...)` constructions in `tests/test_agent.py` to pass `cwd=tmp_path`.

Update `tests/test_cli_run.py` monkeypatch helper:

```python
def fail_task(task, prior_messages, mode):
    raise ConfigError("missing config")
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
C:\ProgramData\anaconda3\python.exe -m pytest tests/test_agent.py tests/test_cli_run.py tests/test_context.py -q --basetemp=.pytest-agent-context
```

Expected: PASS.

Commit:

```bash
git add src/coding_agent/agent.py src/coding_agent/cli.py tests/test_agent.py tests/test_cli_run.py
git commit -m "feat: route agent through context manager"
```

---

### Task 6: Session Recovery

**Files:**
- Modify: `src/coding_agent/session.py`
- Modify: `src/coding_agent/cli.py`
- Test: `tests/test_agent.py`
- Test: `tests/test_cli_run.py`

- [ ] **Step 1: Add session store tests**

Add to `tests/test_agent.py`:

```python
def test_session_store_loads_saved_records(tmp_path):
    records = [{"role": "user", "content": "hello"}]
    store = SessionStore(tmp_path)
    path = store.save(records)

    assert store.load(path) == records


def test_session_store_loads_latest_session(tmp_path):
    store = SessionStore(tmp_path)
    first = store.save([{"role": "user", "content": "first"}])
    second = store.save([{"role": "user", "content": "second"}])

    assert store.latest() in {first, second}
    assert store.load_latest() is not None
```

- [ ] **Step 2: Implement load/latest**

In `src/coding_agent/session.py`:

```python
    def load(self, path: Path | str) -> list[dict[str, Any]]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Session file must contain a message list")
        return data

    def latest(self) -> Path | None:
        if not self.sessions_dir.exists():
            return None
        sessions = sorted(self.sessions_dir.glob("*.json"))
        return sessions[-1] if sessions else None

    def load_latest(self) -> list[dict[str, Any]] | None:
        path = self.latest()
        if path is None:
            return None
        return self.load(path)
```

- [ ] **Step 3: Add chat resume options**

In `src/coding_agent/cli.py`, update command:

```python
@app.command()
def chat(
    resume: Path | None = typer.Option(None, "--resume", help="Resume messages from a session JSON file"),
    resume_latest: bool = typer.Option(False, "--resume-latest", help="Resume the latest saved session"),
) -> None:
```

Initialize messages:

```python
messages: list[dict[str, Any]] = []
store = SessionStore(Path.cwd())
try:
    if resume is not None:
        messages = store.load(resume)
    elif resume_latest:
        messages = store.load_latest() or []
except (OSError, ValueError, json.JSONDecodeError) as exc:
    console.print(f"[red]Error:[/] failed to load session: {exc}")
    raise typer.Exit(1) from exc
```

Import `json`.

- [ ] **Step 4: Add CLI resume smoke test**

Add to `tests/test_cli_run.py`:

```python
import json


def test_chat_resume_reports_loaded_message_count(tmp_path, monkeypatch):
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps([{"role": "user", "content": "old"}]), encoding="utf-8")
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["chat", "--resume", str(session_path)], input="/status\n/exit\n")

    assert result.exit_code == 0
    assert "Messages: 1" in result.output
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
C:\ProgramData\anaconda3\python.exe -m pytest tests/test_agent.py tests/test_cli_run.py -q --basetemp=.pytest-session-recovery
```

Expected: PASS.

Commit:

```bash
git add src/coding_agent/session.py src/coding_agent/cli.py tests/test_agent.py tests/test_cli_run.py
git commit -m "feat: add session recovery"
```

---

### Task 7: Usage Stats and Cache Display

**Files:**
- Modify: `src/coding_agent/context.py`
- Modify: `src/coding_agent/llm.py`
- Modify: `src/coding_agent/agent.py`
- Modify: `src/coding_agent/cli.py`
- Test: `tests/test_context.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Add usage parsing tests**

Add to `tests/test_context.py`:

```python
from coding_agent.context import UsageStats


def test_usage_stats_parses_openai_cached_tokens():
    stats = UsageStats.from_response_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 80},
        }
    )

    assert stats.input_tokens == 100
    assert stats.output_tokens == 20
    assert stats.cached_tokens == 80
    assert stats.cache_hit_ratio == 0.8


def test_usage_stats_parses_input_output_tokens_shape():
    stats = UsageStats.from_response_usage({"input_tokens": 10, "output_tokens": 5, "cached_tokens": 2})

    assert stats.input_tokens == 10
    assert stats.output_tokens == 5
    assert stats.cached_tokens == 2
```

- [ ] **Step 2: Implement UsageStats**

In `src/coding_agent/context.py`:

```python
@dataclass(frozen=True)
class UsageStats:
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None

    @property
    def cache_hit_ratio(self) -> float | None:
        if not self.input_tokens or self.cached_tokens is None:
            return None
        return self.cached_tokens / self.input_tokens

    @classmethod
    def from_response_usage(cls, usage: dict[str, Any] | None) -> "UsageStats":
        usage = usage or {}
        details = usage.get("prompt_tokens_details") or {}
        return cls(
            input_tokens=usage.get("prompt_tokens") or usage.get("input_tokens"),
            output_tokens=usage.get("completion_tokens") or usage.get("output_tokens"),
            cached_tokens=details.get("cached_tokens") if isinstance(details, dict) else usage.get("cached_tokens"),
        )
```

- [ ] **Step 3: Return usage from LLM client**

In `src/coding_agent/llm.py`:

```python
from coding_agent.context import UsageStats
...
return {
    "message": data["choices"][0]["message"],
    "usage": UsageStats.from_response_usage(data.get("usage")),
}
```

- [ ] **Step 4: Store usage in AgentResult**

In `src/coding_agent/agent.py`:

```python
from coding_agent.context import UsageStats

@dataclass
class AgentResult:
    final_answer: str
    messages: list[dict[str, Any]]
    reached_max_steps: bool = False
    usage: UsageStats | None = None
```

When final answer returns:

```python
usage = response.get("usage")
return AgentResult(..., usage=usage)
```

For max steps, keep the last usage seen.

- [ ] **Step 5: Display cache stats in CLI**

Add helper in `src/coding_agent/cli.py`:

```python
def _print_cache_stats(result: AgentResult, enabled: bool) -> None:
    if not enabled or result.usage is None:
        return
    ratio = result.usage.cache_hit_ratio
    if ratio is None:
        return
    console.print(f"Cache: {ratio:.0%} cached input tokens")
```

Call it after final answer in `run` and `chat`, using `config.context.show_cache_stats`. If `_run_task` currently returns only `(result, session_path)`, either return config too or move printing into `_run_task` result metadata. Prefer:

```python
@dataclass(frozen=True)
class RunTaskResult:
    result: AgentResult
    session_path: Path
    show_cache_stats: bool
```

Keep change minimal.

- [ ] **Step 6: Update tests**

Update `tests/test_agent.py::test_llm_client_posts_openai_compatible_chat_request` to expect a `UsageStats` object when usage is present. Add usage to fake response JSON:

```python
json={
    "choices": [{"message": {"role": "assistant", "content": "hello"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 5}},
}
```

Assert:

```python
assert result["usage"].cached_tokens == 5
```

- [ ] **Step 7: Run tests and commit**

Run:

```bash
C:\ProgramData\anaconda3\python.exe -m pytest tests/test_context.py tests/test_agent.py tests/test_cli_run.py -q --basetemp=.pytest-usage-stats
```

Expected: PASS.

Commit:

```bash
git add src/coding_agent/context.py src/coding_agent/llm.py src/coding_agent/agent.py src/coding_agent/cli.py tests/test_context.py tests/test_agent.py
git commit -m "feat: add usage cache stats"
```

---

### Task 8: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/plans/2026-05-13-context-management-design.md` only if implementation diverged.

- [ ] **Step 1: Update README context section**

Add a README section:

```markdown
## Context management

The agent builds each model request from three layers:

1. Stable prefix: agent behavior, safety rules, and tool-use contract.
2. Workspace context: cwd, repo root, branch, git status, recent commits, file tree, and clipped project docs.
3. Dynamic context: recent conversation messages and the current task.

The stable prefix has a local `prompt_cache_key` and is kept byte-stable when tools and rules do not change. Workspace context is sampled each turn and rebuilt only when its fingerprint changes. Recent messages are trimmed with an approximate token budget.
```

Document `[context]` config and `chat --resume/--resume-latest`.

- [ ] **Step 2: Run full test suite**

Run:

```bash
C:\ProgramData\anaconda3\python.exe -m pytest -q --basetemp=.pytest-context-final
```

Expected: PASS.

- [ ] **Step 3: Run import smoke**

Run:

```powershell
$env:PYTHONPATH='src'; C:\ProgramData\anaconda3\python.exe -c "from coding_agent.context import StablePrefixManager, WorkspaceContext; print('context ok')"
```

Expected:

```text
context ok
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/plans/2026-05-13-context-management-design.md
git commit -m "docs: document context management"
```

---

## Self-Review

Spec coverage:

- StablePrefixState and prompt cache key: Task 3.
- WorkspaceContext and workspace fingerprint: Task 2 and Task 3.
- PromptBuilder message order: Task 3.
- Approximate token budget and tool-result truncation: Task 4.
- AgentLoop integration: Task 5.
- Session recovery: Task 6.
- Usage/cache stats parsing and display: Task 7.
- README and verification: Task 8.

Known non-goals preserved:

- No automatic summarization.
- No long-term memory extraction.
- No Anthropic `cache_control`.
- No real tokenizer dependency.
- No full repository indexing.
