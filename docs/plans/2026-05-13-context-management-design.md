# Context Management Design

## Goal

Add a context management module that gives the coding agent a stable, cache-friendly prompt prefix and a small, useful workspace baseline before each model request. The module should help the model understand where it is in a repository without preloading the whole codebase.

The design targets three first-version capabilities:

- Prefix reuse for model-side prompt caching.
- Repository baseline injection for first-step orientation.
- Approximate-token budget trimming for recent conversation history.

Session recovery and usage/cache statistics should have foundations in this design, but long-term memory extraction and automatic summarization are not part of the first implementation.

## Current Problem

The current `AgentLoop` sends a minimal system prompt and then appends the current user task to prior messages. In `chat`, the full message history grows without budget management. The model does not automatically receive a repository baseline such as current directory, repository root, branch, dirty state, recent commits, or high-value project documents.

Short user requests like "fix the tests" or "add an endpoint" are therefore under-specified for the model. A human can infer what to inspect first, but the model benefits from a cheap, stable workspace summary before it chooses tools.

## Selected Approach

Use a two-level prefix design:

1. `StablePrefixState`: global agent rules and tool contract. This block is as stable as possible to maximize implicit model-side prompt cache hits.
2. `WorkspacePrefixState`: repository baseline. This block is sampled before each model request and rebuilt only when its fingerprint changes.

Dynamic information such as user task, recent messages, and future session summaries stays outside both prefix states.

## Core Data Objects

### StablePrefixState

```python
@dataclass(frozen=True)
class StablePrefixState:
    text: str
    system_hash: str
    tool_signature: str
    rules_hash: str
    prompt_cache_key: str
    prompt_version: str
```

`StablePrefixState.text` contains only stable behavior rules:

- Agent identity.
- Operating principles.
- Safety contract.
- Tool-use contract.
- Response contract.

It must not contain current date, cwd, git state, user task, history, or tool results.

`prompt_cache_key` is:

```text
hash(prompt_version + system_hash + tool_signature + rules_hash)
```

### WorkspaceContext

```python
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
```

This is the raw repository baseline.

### WorkspacePrefixState

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
```

`workspace_fingerprint` changes when repository orientation changes: cwd, repo root, branch, status, recent commits, project docs, or file tree.

### ContextEnvelope

```python
@dataclass(frozen=True)
class ContextEnvelope:
    stable_prefix: StablePrefixState
    workspace_prefix: WorkspacePrefixState
    session_summary: str | None
    recent_messages: list[dict[str, Any]]
    current_task: str
    full_context_key: str
```

`full_context_key` is:

```text
hash(
  stable_prefix.prompt_cache_key
  + workspace_prefix.workspace_fingerprint
  + session_summary_hash
  + recent_messages_hash
  + current_task_hash
)
```

This key is for local logging, debugging, and observability. It is not sent to OpenAI-compatible providers for implicit prompt caching.

## Workspace Baseline

The workspace baseline answers the questions the model needs before its first tool call:

- What directory am I in?
- Is this a git repository?
- What is the repository root?
- What branch am I on?
- What is the default branch?
- Is the working tree dirty?
- What changed recently?
- Which high-value project documents should guide behavior?

### Git Fields

Collection rules:

- `cwd`: CLI current working directory.
- `repo_root`: `git rev-parse --show-toplevel`; `None` if not a git repo.
- `branch`: `git branch --show-current`; `None` if unavailable.
- `default_branch`: first try `refs/remotes/origin/HEAD`; then fall back to existing `main` or `master`.
- `status`: `git status --short`; empty outside git repos.
- `recent_commits`: `git log --oneline -5`; empty outside git repos.

Git command failures should not block the agent. They produce empty fields or `repo_root=None`.

### Project Docs

Whitelist:

```python
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
```

Scan both `repo_root` and `cwd`:

```python
for base in (repo_root, cwd):
    for name in DOC_NAMES:
        ...
```

This keeps root-level guidance and subdirectory-local guidance available. Duplicate paths are skipped. Each document is read as UTF-8 with `errors="replace"` and clipped to `doc_max_chars`, default `1200`.

### File Tree

The file tree is a deterministic, clipped navigation aid, not a repository index.

Defaults:

- `tree_max_entries = 200`
- Sorted relative POSIX paths.
- Include files, not directories.

Ignored directories:

```python
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
```

Also ignore directories whose names start with `.pytest`.

## Prefix Text Format

### Stable Prefix

```text
<coding_agent_prefix version="context-v1">
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
</coding_agent_prefix>
```

### Workspace Prefix

```text
<workspace_context>
cwd: ...
repo_root: ...
is_git_repo: true
branch: main
default_branch: main

git_status:
  clean

recent_commits:
  5e424eb fix: execute allowed commands without shell

file_tree:
  README.md
  pyproject.toml
  src/coding_agent/agent.py

project_docs:
  <doc path="README.md">
  ...
  </doc>
</workspace_context>
```

If output is clipped, render a deterministic truncation marker such as:

```text
... [truncated: showing first 200 entries]
```

## Prompt Message Order

`PromptBuilder` creates messages in this order:

```python
[
    {"role": "system", "content": stable_prefix.text},
    {"role": "user", "content": workspace_prefix.text},
    {"role": "user", "content": session_summary_text},  # future, only when present
    *trimmed_recent_messages,
    {"role": "user", "content": current_task_text},
]
```

Current task text:

```text
<current_task>
mode: run|chat
content:
...
</current_task>
```

`AgentLoop` should no longer manually inject a system prompt. It should receive messages from the context manager.

## Approximate Token Budget

Use a model-independent token estimate:

```python
estimated_tokens = ceil(len(text) / 4)
```

Default context config:

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

Budget rules:

1. Keep stable prefix.
2. Keep workspace prefix.
3. Keep current task.
4. Keep recent messages from newest to oldest until the recent-message budget or total budget is reached.
5. If a single tool message is too long, truncate its `content` while preserving `role`, `name`, and `tool_call_id`.

No real tokenizer, automatic summary, or semantic retrieval in the first version.

## Session Recovery

Add basic session loading but avoid long-term memory extraction.

SessionStore should support:

- `load(path) -> list[dict[str, Any]]`
- `latest() -> Path | None`
- `load_latest() -> list[dict[str, Any]] | None`

CLI can support:

- `coding-agent chat --resume PATH`
- `coding-agent chat --resume-latest`

Recovered messages still pass through `MessageBudget`; long sessions are not blindly inserted into the prompt.

## Usage and Cache Observability

Define:

```python
@dataclass(frozen=True)
class UsageStats:
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
```

OpenAI-compatible parsing should support:

- `usage.prompt_tokens`
- `usage.completion_tokens`
- `usage.prompt_tokens_details.cached_tokens`
- `usage.input_tokens`
- `usage.output_tokens`
- `usage.cached_tokens`

CLI display:

```text
Cache: 85% cached input tokens
```

Only display cache stats when `cached_tokens` and input token count are available. Do not estimate dollar savings in the first version.

## Error Handling

- Git command failure: produce empty git fields.
- Non-git directory: `repo_root=None`, `is_git_repo=false`.
- Document read failure: skip the document.
- Non-UTF-8 document: read with replacement.
- Large file tree or document: clip deterministically.
- Fingerprint calculation failure: rebuild workspace prefix instead of blocking the request.
- Session load failure: surface a clear CLI error.

## Testing

Add `tests/test_context.py`:

- Stable prefix key is stable for the same tool schemas.
- Tool schema changes alter `tool_signature`.
- Non-git workspace builds a valid workspace context.
- Git workspace captures repo root, branch, status, and recent commits.
- Project docs scan both repo root and cwd and deduplicate paths.
- Project docs are clipped to `doc_max_chars`.
- File tree ignores `.git`, `.coding-agent`, `__pycache__`, `.pytest*`, `node_modules`, `dist`, and `build`.
- Workspace fingerprint changes when git status changes.
- PromptBuilder message order is stable.
- MessageBudget preserves stable prefix, workspace prefix, and current task.
- MessageBudget drops oldest recent messages first.
- Tool message content is truncated while preserving tool metadata.
- SessionStore loads explicit and latest session files.
- UsageStats parses cached token fields from model responses.

Update existing tests:

- `test_agent.py`: AgentLoop still handles tool-call then final-answer flow through ContextManager.
- `test_cli_run.py`: chat resume path and error handling.
- `test_config.py`: missing `[context]` uses defaults; explicit `[context]` overrides defaults.

## Non-Goals

- Automatic summary compression.
- Long-term memory or fact extraction.
- Anthropic explicit `cache_control`.
- Real tokenizer integration.
- Full repository indexing.
- Semantic retrieval.
