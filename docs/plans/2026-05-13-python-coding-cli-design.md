# Python Coding CLI Design

## Goal

Build a Python-based AI coding assistant CLI. The first version should behave like a practical local coding agent: it can read and modify files inside the current project, run approved shell commands, call any OpenAI-compatible model endpoint, and work in both interactive and one-shot modes.

## Selected Approach

Use a lightweight custom agent core rather than a large external agent framework.

This keeps the security model, tool protocol, logging, and CLI behavior explicit. The first version should be usable quickly, while leaving room for later improvements such as context compaction, richer planning, and additional model providers.

## CLI Surface

The package installs a command named `coding-agent`.

Initial commands:

```bash
coding-agent init
coding-agent chat
coding-agent run "fix the failing tests"
```

`init` creates project-local configuration. `chat` starts a persistent REPL session. `run` executes one task and exits.

## Configuration

Project configuration lives at `.coding-agent/config.toml`.

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
```

The command policy is configurable. Deny rules take precedence over allow rules. Filesystem access is limited to the current project directory in the first version.

## Architecture

The codebase is split into small modules:

- `cli`: Typer command entrypoints for `init`, `chat`, and `run`.
- `config`: Load, merge, and validate configuration.
- `agent`: Main agent loop and prompt construction.
- `llm`: OpenAI-compatible chat client.
- `tools`: Tool implementations and schema registration.
- `sandbox`: Workspace path validation and filesystem boundaries.
- `policy`: Shell command allow and deny checks.
- `session`: Execution logs and conversation persistence.

## Components

`ConfigLoader` loads defaults and project configuration, then validates required model and agent fields.

`OpenAICompatibleClient` wraps a Chat Completions-compatible endpoint. The first version targets broad compatibility by using `base_url`, `api_key_env`, and `model`.

`AgentLoop` owns message history, calls the model, dispatches tool calls, appends tool results, and stops on final answer, max steps, or unrecoverable error.

`ToolRegistry` registers tools and exposes schemas to the model.

`WorkspaceSandbox` normalizes paths and rejects reads or writes outside the project root.

`CommandPolicy` evaluates shell commands against configured deny and allow rules.

`ShellRunner` executes only allowed commands and captures stdout, stderr, exit code, and timeout status.

`SessionStore` writes task logs under `.coding-agent/sessions/`.

## Initial Tools

The first version exposes:

- `list_files(path)`
- `read_file(path)`
- `search_text(query, path=".")`
- `write_file(path, content)`
- `apply_patch(patch)`
- `run_shell(command)`

The system prompt should prefer `apply_patch` for modifying existing files. `write_file` remains available for creating new files or replacing generated files intentionally.

## Run Mode Data Flow

1. Parse the CLI task and current working directory.
2. Load `.coding-agent/config.toml`; if missing, tell the user to run `coding-agent init`.
3. Create the sandbox, command policy, tool registry, model client, and session store.
4. Build system messages describing tools, workspace boundaries, shell policy, output rules, and max steps.
5. Call the model.
6. If the model requests a tool call, validate it through sandbox or policy, execute it, and append the result to history.
7. Repeat until the model returns a final answer, max steps are reached, or an unrecoverable error occurs.
8. Save a session log.

## Chat Mode Data Flow

`chat` starts a REPL and keeps conversation state across turns. Each user input creates a new task turn while preserving useful context.

Initial slash commands:

- `/exit`
- `/clear`
- `/status`

Additional commands such as `/model` and `/config` can be added later.

## Error Handling

Configuration missing: show a clear message to run `coding-agent init`.

API key missing: name the missing environment variable.

Model request failure: display a concise HTTP or network error and save details in the session log.

Invalid tool arguments: return the validation error to the model so it can correct the next step.

Path outside workspace: reject the operation and record a security event.

Denied shell command: reject immediately.

Shell command not allowed: reject and tell the user to update the allow list.

Shell timeout: terminate the command and return a timeout result.

Max steps reached: stop and summarize current progress.

## Testing

The first test suite focuses on deterministic core behavior:

- `WorkspaceSandbox`: legal project paths, `..` traversal, absolute paths, and project boundary checks.
- `CommandPolicy`: allow, deny, deny precedence, exact and prefix matching.
- `ConfigLoader`: default configuration, project overrides, and missing fields.
- `ToolRegistry`: registration, schema output, and unknown tool handling.
- `AgentLoop`: fake model sequences for tool calls and final answers.
- CLI smoke tests: `init` creates config, and `run` reports missing API key clearly.

## First-Version Non-Goals

- Multi-agent execution.
- Long-term memory.
- Automatic context compaction.
- Remote repository management.
- Full TUI interface.
- Executing shell commands outside configured policy.
- Reading or writing files outside the current project directory.
