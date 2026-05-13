# coding-agent

`coding-agent` is a Python CLI for running a local AI coding assistant against a
project workspace. It uses an OpenAI-compatible chat completions endpoint, gives
the model a small set of file and shell tools, and keeps file access and command
execution inside explicit project-level boundaries.

This repository is an MVP. The implementation is intentionally small so the
agent loop, tool protocol, configuration, command policy, and safety checks are
easy to inspect.

## Current MVP capabilities

- Installable command: `coding-agent`.
- Project-local configuration in `.coding-agent/config.toml`.
- One-shot task mode with `coding-agent run "..."`.
- Interactive REPL mode with `coding-agent chat`.
- Chat resume from a saved session with `coding-agent chat --resume PATH` or
  `coding-agent chat --resume-latest`.
- OpenAI-compatible model calls through `/chat/completions`.
- Tool calls for listing files, reading UTF-8 files, writing UTF-8 files,
  searching text, and running allowed commands.
- Workspace sandboxing for all file tool paths.
- Configurable allow and deny lists for shell commands.
- Context management with a stable prompt prefix, workspace baseline, and
  budgeted recent conversation history.
- Sanitized session logs written to `.coding-agent/sessions/`.

## Project structure

```text
src/coding_agent/
  agent.py      Main agent loop and tool-call handling.
  cli.py        Typer commands: init, run, and chat.
  config.py     TOML config loading, validation, and environment lookup.
  context.py    Prompt prefixes, workspace context, message budgeting, usage stats.
  llm.py        OpenAI-compatible chat completions client.
  policy.py     Shell command allow/deny policy.
  sandbox.py    Workspace path resolution and boundary checks.
  session.py    JSON session log writer and resume loader.
  shell.py      Policy-gated subprocess execution.
  tools.py      Tool registry and built-in tool implementations.

tests/          Unit and CLI smoke tests for the MVP behavior.
docs/plans/     Design notes and implementation plans.
```

## How the agent works

1. `coding-agent run` or `coding-agent chat` loads
   `.coding-agent/config.toml` from the current working directory.
2. The CLI builds a `WorkspaceSandbox`, `CommandPolicy`, `ShellRunner`,
   `ToolRegistry`, model client, and `AgentLoop`.
3. The context manager builds each request from a stable system prefix, a
   workspace context block, budgeted recent messages, and the current task.
4. The agent sends that prompt and the tool schemas to the configured
   OpenAI-compatible endpoint.
5. If the model returns tool calls, the agent validates and dispatches them
   through the local tool registry.
6. Tool results are appended to the reusable conversation history as tool
   messages.
7. The loop continues until the model returns a final answer or `max_steps` is
   reached.
8. Sanitized conversation history is saved as JSON under
   `.coding-agent/sessions/`. Generated prompt blocks such as workspace context
   and current task wrappers are not saved.

## Context management

Each model request is assembled in this order:

1. Stable system prefix: behavior, safety, tool, and response rules. This block
   has a local `prompt_cache_key` and stays byte-stable while the prompt version
   and tool schemas do not change.
2. Workspace context: current directory, repository root, branch, default
   branch, git status, recent commits, a clipped file tree, and clipped project
   docs such as `AGENTS.md`, `README.md`, `pyproject.toml`, and `package.json`.
   This block has a fingerprint so it can be reused until workspace inputs
   change.
3. Dynamic context: recent saved conversation messages and the current task.
   The full context key also includes the run mode (`run` or `chat`).

Recent conversation history is trimmed with an approximate token budget. Oldest
messages are dropped first, assistant tool-call groups stay together with their
tool results, and oversized tool-result content is truncated while preserving
tool metadata.

## Configuration

Run `coding-agent init` to create `.coding-agent/config.toml`.

Example:

```toml
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
allow = [
  "python -m pytest",
  "pytest",
  "ruff",
  "mypy",
  "git status",
  "git diff"
]
deny = [
  "rm",
  "del",
  "rmdir",
  "git reset",
  "git checkout",
  "powershell Remove-Item"
]

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

Meaning:

- `model.base_url`: Base URL for an OpenAI-compatible API. The client posts to
  `{base_url}/chat/completions`.
- `model.api_key_env`: Environment variable that contains the API key.
- `model.model`: Model name sent in the chat completions request.
- `agent.max_steps`: Maximum model/tool iterations before the agent stops.
- `agent.stream`: Reserved configuration flag. The current client does not
  stream responses.
- `workspace.root`: Root directory for file tools and shell command working
  directory. Relative paths are resolved from the project root.
- `commands.allow`: Exact command or command-prefix rules that may run.
- `commands.deny`: Exact command or command-prefix rules that are blocked before
  allow rules are considered.
- `context.max_input_tokens`: Reserved for total input budgeting. Current
  trimming is controlled by `context.recent_message_tokens`.
- `context.reserved_output_tokens`: Reserved output budget for future total
  context calculations.
- `context.recent_message_tokens`: Approximate-token budget for reusable recent
  conversation messages.
- `context.project_context_tokens`: Reserved for workspace/project context
  budgeting.
- `context.doc_max_chars`: Maximum characters read from each whitelisted project
  document.
- `context.tree_max_entries`: Maximum file paths included in the workspace file
  tree.
- `context.include_project_docs`: Include clipped high-value project docs in the
  workspace context.
- `context.include_file_tree`: Include a clipped file tree in the workspace
  context.
- `context.include_git_status`: Include `git status --short` output when inside
  a git repository.
- `context.include_recent_commits`: Include up to five recent commits when
  inside a git repository.
- `context.restore_last_session`: Reserved flag. Use `chat --resume-latest` to
  resume explicitly.
- `context.show_cache_stats`: Print cache hit percentage when the model response
  includes both input token count and cached input token count.

Set the configured API key before running the agent:

```bash
export OPENAI_API_KEY="..."
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "..."
```

## Commands

### `coding-agent init`

Creates `.coding-agent/config.toml` in the target project.

```bash
coding-agent init
coding-agent init --path /path/to/project
```

If the config already exists, the command leaves it in place.

### `coding-agent run`

Runs one task and exits.

```bash
coding-agent run "inspect the project and summarize it"
coding-agent run "run the tests and explain the failures"
```

The command prints the final answer and the path to the saved session log.

### `coding-agent chat`

Starts an interactive session. Conversation state is kept across turns until it
is cleared or the process exits.

```bash
coding-agent chat
coding-agent chat --resume .coding-agent/sessions/20260513-120000-000000.json
coding-agent chat --resume-latest
```

Slash commands:

- `/exit`: Quit chat mode.
- `/clear`: Clear the current conversation history.
- `/status`: Print the number of messages currently in memory.

`--resume PATH` loads validated conversation messages from a specific saved
session. `--resume-latest` loads the newest JSON session under
`.coding-agent/sessions/` for the current working directory. Resume options
restore sanitized conversation history only; generated prompt blocks are
recreated for the current workspace on each request.

## Usage and cache stats

The OpenAI-compatible client records usage metadata when the provider returns
it. The CLI prints a line such as:

```text
Cache: 85% cached input tokens
```

This display depends on `context.show_cache_stats = true` and a response usage
payload that includes both input tokens and cached input tokens. Supported fields
include OpenAI-style `prompt_tokens` with
`prompt_tokens_details.cached_tokens`, and compatible `input_tokens` plus
`cached_tokens`. If those fields are absent, no cache line is printed.

## Safety model

### Workspace sandbox

All file tools resolve paths through `WorkspaceSandbox`. Relative paths are
resolved under `workspace.root`; absolute paths must still be inside that root.
Parent traversal such as `../outside.txt` is rejected. Tool results use
workspace-relative POSIX-style paths.

The built-in file tools are:

- `list_files(path=".")`
- `read_file(path)`
- `write_file(path, content)`
- `search_text(query, path=".")`

`write_file` can create parent directories and writes UTF-8 text.

### Command policy

Shell commands go through `CommandPolicy` before execution.

- Deny rules are checked first and take precedence.
- Allow rules are checked after deny rules.
- Rules match either the full command or a prefix followed by a space.
- Commands not matched by the allow list are rejected.
- Shell control operators are rejected before allow matching.

Rejected control syntax includes newlines, `&&`, `||`, `;`, `&`, `|`, `>`, `<`,
command substitution with `$()`, and backticks.

### Shell execution constraints

Allowed commands are parsed into an argv list and executed with
`subprocess.run(..., shell=False)`. The runner captures stdout, stderr, exit
code, and timeout status. The default timeout is 120 seconds.

The runner resolves the executable from `PATH` and refuses to execute a
workspace-local executable. This prevents a project file such as `pytest.bat`
from being run just because a bare command like `pytest` is allowed.

## Development setup

Requirements:

- Python 3.11 or newer.

Install the package in editable mode with test dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
python -m pytest -q
```

Useful verification commands for this repository:

```bash
python -m pytest -q --basetemp=.pytest-readme-agent
git diff -- README.md
```

## Current limitations and non-goals

- `apply_patch` is registered as a tool name but is explicitly unsupported in
  the MVP and always returns an error.
- Streaming is configured but not implemented by the model client.
- There is no automatic summary compression, long-term memory, or fact
  extraction.
- Context budgeting uses an approximate token estimate, not a model tokenizer.
- The workspace context is a clipped orientation aid, not a full repository
  index or semantic retrieval system.
- There is no multi-agent execution.
- There is no full-screen TUI.
- The CLI only loads `.coding-agent/config.toml` from the current working
  directory.
- File access outside `workspace.root` is not supported.
- Shell commands outside the configured policy are not supported.
