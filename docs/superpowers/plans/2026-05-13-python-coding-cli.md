# Python Coding CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python AI coding assistant CLI with `init`, `chat`, and `run`, project-local file sandboxing, configurable shell command policy, OpenAI-compatible model calls, and session logging.

**Architecture:** Implement a small custom agent core instead of a large agent framework. Keep model access, tools, sandboxing, command policy, session logging, and CLI entrypoints as separate modules with deterministic unit tests around safety-critical behavior.

**Tech Stack:** Python 3.11+, Typer, Rich, httpx, pydantic, pytest.

---

## File Structure

- Create `pyproject.toml`: package metadata, console script, dependencies, pytest config.
- Create `src/coding_agent/__init__.py`: package marker and version.
- Create `src/coding_agent/cli.py`: Typer app with `init`, `run`, and `chat`.
- Create `src/coding_agent/config.py`: config dataclasses and TOML loading.
- Create `src/coding_agent/sandbox.py`: workspace path validation.
- Create `src/coding_agent/policy.py`: shell allow/deny matching.
- Create `src/coding_agent/shell.py`: subprocess execution wrapper.
- Create `src/coding_agent/tools.py`: tool registry and built-in tools.
- Create `src/coding_agent/llm.py`: OpenAI-compatible chat client.
- Create `src/coding_agent/session.py`: session log writer.
- Create `src/coding_agent/agent.py`: agent loop orchestration.
- Create `tests/`: focused pytest coverage for config, sandbox, policy, tools, agent loop, and CLI smoke behavior.

---

### Task 1: Package Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/coding_agent/__init__.py`
- Create: `src/coding_agent/cli.py`
- Test: `tests/test_cli_init.py`

- [ ] **Step 1: Create package metadata**

Write `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "coding-agent"
version = "0.1.0"
description = "A Python AI coding assistant CLI"
requires-python = ">=3.11"
dependencies = [
  "httpx>=0.27",
  "pydantic>=2.7",
  "rich>=13.7",
  "typer>=0.12",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2",
  "pytest-cov>=5.0",
]

[project.scripts]
coding-agent = "coding_agent.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 2: Create minimal package files**

Write `src/coding_agent/__init__.py`:

```python
__version__ = "0.1.0"
```

Write initial `src/coding_agent/cli.py`:

```python
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="AI coding assistant CLI")
console = Console()


@app.command()
def init(path: Path = typer.Option(Path("."), "--path", "-p", help="Project root")) -> None:
    """Create a project-local coding-agent config."""
    config_dir = path / ".coding-agent"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"
    if config_path.exists():
        console.print(f"[yellow]Config already exists:[/] {config_path}")
        return
    config_path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    console.print(f"[green]Created config:[/] {config_path}")


@app.command()
def run(task: str) -> None:
    """Run one coding task and exit."""
    console.print(f"Task: {task}")
    raise typer.Exit(1)


@app.command()
def chat() -> None:
    """Start an interactive coding session."""
    console.print("Chat mode is not wired yet.")
    raise typer.Exit(1)


DEFAULT_CONFIG = """[model]
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
"""
```

- [ ] **Step 3: Write CLI init smoke test**

Write `tests/test_cli_init.py`:

```python
from typer.testing import CliRunner

from coding_agent.cli import app


def test_init_creates_project_config(tmp_path):
    runner = CliRunner()

    result = runner.invoke(app, ["init", "--path", str(tmp_path)])

    assert result.exit_code == 0
    config_path = tmp_path / ".coding-agent" / "config.toml"
    assert config_path.exists()
    assert "base_url" in config_path.read_text(encoding="utf-8")
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_cli_init.py -v`

Expected: PASS.

- [ ] **Step 5: Commit if git exists**

Run only in a git repository:

```bash
git add pyproject.toml src/coding_agent tests/test_cli_init.py
git commit -m "feat: scaffold coding agent cli"
```

---

### Task 2: Configuration Loader

**Files:**
- Create: `src/coding_agent/config.py`
- Modify: `src/coding_agent/cli.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Write `tests/test_config.py`:

```python
import os

import pytest

from coding_agent.config import ConfigError, load_config


def test_load_config_reads_project_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_dir = tmp_path / ".coding-agent"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        """
[model]
base_url = "http://localhost:1234/v1"
api_key_env = "OPENAI_API_KEY"
model = "local-model"

[agent]
max_steps = 3
stream = false

[workspace]
root = "."

[commands]
allow = ["pytest"]
deny = ["rm"]
""",
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config.model.base_url == "http://localhost:1234/v1"
    assert config.model.api_key == "test-key"
    assert config.agent.max_steps == 3
    assert config.commands.allow == ["pytest"]


def test_load_config_reports_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="coding-agent init"):
        load_config(tmp_path)


def test_load_config_reports_missing_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        load_config(tmp_path)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL because `coding_agent.config` does not exist.

- [ ] **Step 3: Implement config loader**

Write `src/coding_agent/config.py`:

```python
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    api_key_env: str
    api_key: str
    model: str


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int
    stream: bool


@dataclass(frozen=True)
class WorkspaceConfig:
    root: Path


@dataclass(frozen=True)
class CommandConfig:
    allow: list[str]
    deny: list[str]


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    model: ModelConfig
    agent: AgentConfig
    workspace: WorkspaceConfig
    commands: CommandConfig


def load_config(project_root: Path) -> AppConfig:
    project_root = project_root.resolve()
    config_path = project_root / ".coding-agent" / "config.toml"
    if not config_path.exists():
        raise ConfigError("Missing .coding-agent/config.toml. Run coding-agent init first.")

    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    try:
        model_data = data["model"]
        agent_data = data["agent"]
        workspace_data = data["workspace"]
        command_data = data["commands"]
        api_key_env = str(model_data["api_key_env"])
        api_key = os.environ[api_key_env]
    except KeyError as exc:
        missing = exc.args[0]
        raise ConfigError(f"Missing required configuration or environment value: {missing}") from exc

    workspace_root = (project_root / str(workspace_data.get("root", "."))).resolve()
    return AppConfig(
        project_root=project_root,
        model=ModelConfig(
            base_url=str(model_data["base_url"]).rstrip("/"),
            api_key_env=api_key_env,
            api_key=api_key,
            model=str(model_data["model"]),
        ),
        agent=AgentConfig(
            max_steps=int(agent_data.get("max_steps", 20)),
            stream=bool(agent_data.get("stream", True)),
        ),
        workspace=WorkspaceConfig(root=workspace_root),
        commands=CommandConfig(
            allow=[str(item) for item in command_data.get("allow", [])],
            deny=[str(item) for item in command_data.get("deny", [])],
        ),
    )
```

- [ ] **Step 4: Run config tests**

Run: `python -m pytest tests/test_config.py -v`

Expected: PASS.

---

### Task 3: Workspace Sandbox

**Files:**
- Create: `src/coding_agent/sandbox.py`
- Test: `tests/test_sandbox.py`

- [ ] **Step 1: Write sandbox tests**

Write `tests/test_sandbox.py`:

```python
import pytest

from coding_agent.sandbox import SandboxError, WorkspaceSandbox


def test_resolve_allows_project_file(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    resolved = sandbox.resolve("src/app.py")
    assert resolved == tmp_path / "src" / "app.py"


def test_resolve_rejects_parent_traversal(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    with pytest.raises(SandboxError, match="outside workspace"):
        sandbox.resolve("../secret.txt")


def test_resolve_rejects_absolute_path_outside_workspace(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    outside = tmp_path.parent / "secret.txt"
    with pytest.raises(SandboxError, match="outside workspace"):
        sandbox.resolve(outside)


def test_relative_path_returns_posix_path(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    assert sandbox.relative_path(tmp_path / "a" / "b.txt") == "a/b.txt"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_sandbox.py -v`

Expected: FAIL because `coding_agent.sandbox` does not exist.

- [ ] **Step 3: Implement sandbox**

Write `src/coding_agent/sandbox.py`:

```python
from __future__ import annotations

from pathlib import Path


class SandboxError(RuntimeError):
    pass


class WorkspaceSandbox:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise SandboxError(f"Path is outside workspace: {path}")
        return resolved

    def relative_path(self, path: str | Path) -> str:
        resolved = self.resolve(path)
        return resolved.relative_to(self.root).as_posix()
```

- [ ] **Step 4: Run sandbox tests**

Run: `python -m pytest tests/test_sandbox.py -v`

Expected: PASS.

---

### Task 4: Command Policy and Shell Runner

**Files:**
- Create: `src/coding_agent/policy.py`
- Create: `src/coding_agent/shell.py`
- Test: `tests/test_policy.py`
- Test: `tests/test_shell.py`

- [ ] **Step 1: Write command policy tests**

Write `tests/test_policy.py`:

```python
from coding_agent.policy import CommandDecision, CommandPolicy


def test_allows_exact_or_prefix_match():
    policy = CommandPolicy(allow=["pytest", "python -m pytest"], deny=[])
    assert policy.evaluate("pytest tests").decision == CommandDecision.ALLOW
    assert policy.evaluate("python -m pytest tests").decision == CommandDecision.ALLOW


def test_denies_exact_or_prefix_match():
    policy = CommandPolicy(allow=["rm"], deny=["rm"])
    result = policy.evaluate("rm -rf build")
    assert result.decision == CommandDecision.DENY
    assert "deny rule" in result.reason


def test_rejects_unlisted_command():
    policy = CommandPolicy(allow=["pytest"], deny=[])
    result = policy.evaluate("python script.py")
    assert result.decision == CommandDecision.REJECT
    assert "not in allow list" in result.reason
```

- [ ] **Step 2: Implement command policy**

Write `src/coding_agent/policy.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommandDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REJECT = "reject"


@dataclass(frozen=True)
class PolicyResult:
    decision: CommandDecision
    reason: str


class CommandPolicy:
    def __init__(self, allow: list[str], deny: list[str]) -> None:
        self.allow = [item.strip() for item in allow if item.strip()]
        self.deny = [item.strip() for item in deny if item.strip()]

    def evaluate(self, command: str) -> PolicyResult:
        normalized = " ".join(command.strip().split())
        for rule in self.deny:
            if _matches(normalized, rule):
                return PolicyResult(CommandDecision.DENY, f"Command matched deny rule: {rule}")
        for rule in self.allow:
            if _matches(normalized, rule):
                return PolicyResult(CommandDecision.ALLOW, f"Command matched allow rule: {rule}")
        return PolicyResult(CommandDecision.REJECT, "Command is not in allow list")


def _matches(command: str, rule: str) -> bool:
    return command == rule or command.startswith(rule + " ")
```

- [ ] **Step 3: Write shell runner tests**

Write `tests/test_shell.py`:

```python
from coding_agent.policy import CommandPolicy
from coding_agent.shell import ShellRunner


def test_shell_runner_rejects_unlisted_command(tmp_path):
    runner = ShellRunner(CommandPolicy(allow=[], deny=[]), tmp_path)

    result = runner.run("python --version")

    assert result.exit_code is None
    assert result.allowed is False
    assert "not in allow list" in result.stderr
```

- [ ] **Step 4: Implement shell runner**

Write `src/coding_agent/shell.py`:

```python
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from coding_agent.policy import CommandDecision, CommandPolicy


@dataclass(frozen=True)
class ShellResult:
    command: str
    allowed: bool
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


class ShellRunner:
    def __init__(self, policy: CommandPolicy, cwd: Path, timeout_seconds: int = 120) -> None:
        self.policy = policy
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds

    def run(self, command: str) -> ShellResult:
        decision = self.policy.evaluate(command)
        if decision.decision is not CommandDecision.ALLOW:
            return ShellResult(command, False, None, "", decision.reason)
        try:
            completed = subprocess.run(
                command,
                cwd=self.cwd,
                shell=True,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return ShellResult(command, True, None, exc.stdout or "", exc.stderr or "Command timed out", True)
        return ShellResult(command, True, completed.returncode, completed.stdout, completed.stderr)
```

- [ ] **Step 5: Run policy and shell tests**

Run: `python -m pytest tests/test_policy.py tests/test_shell.py -v`

Expected: PASS.

---

### Task 5: Tool Registry and Built-In Tools

**Files:**
- Create: `src/coding_agent/tools.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1: Write tool tests**

Write `tests/test_tools.py`:

```python
from coding_agent.policy import CommandPolicy
from coding_agent.sandbox import WorkspaceSandbox
from coding_agent.shell import ShellRunner
from coding_agent.tools import create_default_tools


def test_read_and_write_file_tools(tmp_path):
    registry = create_default_tools(
        WorkspaceSandbox(tmp_path),
        ShellRunner(CommandPolicy(allow=[], deny=[]), tmp_path),
    )

    write_result = registry.call("write_file", {"path": "hello.txt", "content": "hello"})
    read_result = registry.call("read_file", {"path": "hello.txt"})

    assert write_result["ok"] is True
    assert read_result["content"] == "hello"


def test_search_text_finds_matches(tmp_path):
    (tmp_path / "app.py").write_text("print('needle')\n", encoding="utf-8")
    registry = create_default_tools(
        WorkspaceSandbox(tmp_path),
        ShellRunner(CommandPolicy(allow=[], deny=[]), tmp_path),
    )

    result = registry.call("search_text", {"query": "needle", "path": "."})

    assert result["matches"][0]["path"] == "app.py"


def test_unknown_tool_returns_error(tmp_path):
    registry = create_default_tools(
        WorkspaceSandbox(tmp_path),
        ShellRunner(CommandPolicy(allow=[], deny=[]), tmp_path),
    )

    result = registry.call("missing", {})

    assert result["ok"] is False
    assert "Unknown tool" in result["error"]
```

- [ ] **Step 2: Implement tools**

Write `src/coding_agent/tools.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coding_agent.sandbox import WorkspaceSandbox
from coding_agent.shell import ShellRunner

ToolFunc = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: ToolFunc


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool: {name}"}
        try:
            return tool.func(arguments)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def create_default_tools(sandbox: WorkspaceSandbox, shell: ShellRunner) -> ToolRegistry:
    registry = ToolRegistry()

    def list_files(args: dict[str, Any]) -> dict[str, Any]:
        base = sandbox.resolve(args.get("path", "."))
        files = [sandbox.relative_path(path) for path in base.rglob("*") if path.is_file()]
        return {"ok": True, "files": sorted(files)}

    def read_file(args: dict[str, Any]) -> dict[str, Any]:
        path = sandbox.resolve(args["path"])
        return {"ok": True, "path": sandbox.relative_path(path), "content": path.read_text(encoding="utf-8")}

    def write_file(args: dict[str, Any]) -> dict[str, Any]:
        path = sandbox.resolve(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(args["content"]), encoding="utf-8")
        return {"ok": True, "path": sandbox.relative_path(path)}

    def search_text(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        base = sandbox.resolve(args.get("path", "."))
        matches: list[dict[str, Any]] = []
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            try:
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                    if query in line:
                        matches.append({"path": sandbox.relative_path(path), "line": line_number, "text": line})
            except UnicodeDecodeError:
                continue
        return {"ok": True, "matches": matches[:100]}

    def apply_patch(args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": False, "error": "apply_patch is reserved for a later implementation task"}

    def run_shell(args: dict[str, Any]) -> dict[str, Any]:
        result = shell.run(str(args["command"]))
        return {
            "ok": result.allowed,
            "command": result.command,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
        }

    string_path_param = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    registry.register(Tool("list_files", "List files under a workspace path", string_path_param, list_files))
    registry.register(Tool("read_file", "Read a UTF-8 file", string_path_param, read_file))
    registry.register(
        Tool(
            "write_file",
            "Write a UTF-8 file inside the workspace",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            write_file,
        )
    )
    registry.register(
        Tool(
            "search_text",
            "Search text under a workspace path",
            {
                "type": "object",
                "properties": {"query": {"type": "string"}, "path": {"type": "string"}},
                "required": ["query"],
            },
            search_text,
        )
    )
    registry.register(
        Tool(
            "apply_patch",
            "Apply a patch inside the workspace",
            {"type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"]},
            apply_patch,
        )
    )
    registry.register(
        Tool(
            "run_shell",
            "Run an allowed shell command",
            {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            run_shell,
        )
    )
    return registry
```

- [ ] **Step 3: Run tool tests**

Run: `python -m pytest tests/test_tools.py -v`

Expected: PASS.

---

### Task 6: LLM Client, Session Store, and Agent Loop

**Files:**
- Create: `src/coding_agent/llm.py`
- Create: `src/coding_agent/session.py`
- Create: `src/coding_agent/agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write agent loop test with fake LLM**

Write `tests/test_agent.py`:

```python
from coding_agent.agent import AgentLoop
from coding_agent.policy import CommandPolicy
from coding_agent.sandbox import WorkspaceSandbox
from coding_agent.shell import ShellRunner
from coding_agent.tools import create_default_tools


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "write_file", "arguments": "{\"path\":\"x.txt\",\"content\":\"ok\"}"},
                        }
                    ],
                }
            }
        return {"message": {"role": "assistant", "content": "done", "tool_calls": []}}


def test_agent_loop_executes_tool_then_returns_final(tmp_path):
    tools = create_default_tools(
        WorkspaceSandbox(tmp_path),
        ShellRunner(CommandPolicy(allow=[], deny=[]), tmp_path),
    )
    loop = AgentLoop(FakeLLM(), tools, max_steps=3)

    result = loop.run("create x.txt")

    assert result.final_answer == "done"
    assert (tmp_path / "x.txt").read_text(encoding="utf-8") == "ok"
```

- [ ] **Step 2: Implement LLM client**

Write `src/coding_agent/llm.py`:

```python
from __future__ import annotations

from typing import Any

import httpx


class LLMError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(str(exc)) from exc
        data = response.json()
        return {"message": data["choices"][0]["message"]}
```

- [ ] **Step 3: Implement session store**

Write `src/coding_agent/session.py`:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SessionStore:
    def __init__(self, project_root: Path) -> None:
        self.session_dir = project_root / ".coding-agent" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def save(self, records: list[dict[str, Any]]) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.session_dir / f"{timestamp}.json"
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
```

- [ ] **Step 4: Implement agent loop**

Write `src/coding_agent/agent.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from coding_agent.tools import ToolRegistry


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class AgentResult:
    final_answer: str
    messages: list[dict[str, Any]]
    reached_max_steps: bool = False


class AgentLoop:
    def __init__(self, client: ChatClient, tools: ToolRegistry, max_steps: int) -> None:
        self.client = client
        self.tools = tools
        self.max_steps = max_steps

    def run(self, task: str, prior_messages: list[dict[str, Any]] | None = None) -> AgentResult:
        messages = list(prior_messages or [])
        if not messages:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": task})

        for _ in range(self.max_steps):
            response = self.client.chat(messages, self.tools.schemas())
            message = response["message"]
            messages.append(message)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return AgentResult(str(message.get("content") or ""), messages)
            for call in tool_calls:
                name = call["function"]["name"]
                raw_arguments = call["function"].get("arguments") or "{}"
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    result = {"ok": False, "error": f"Invalid JSON arguments: {exc}"}
                else:
                    result = self.tools.call(name, arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        return AgentResult("Stopped after reaching max steps.", messages, reached_max_steps=True)


SYSTEM_PROMPT = """You are a local coding agent.
Work inside the current project only.
Use tools to inspect files before editing.
Prefer apply_patch for editing existing files.
Run shell commands only through run_shell.
Stop when the task is complete and summarize changed files and verification."""
```

- [ ] **Step 5: Run agent test**

Run: `python -m pytest tests/test_agent.py -v`

Expected: PASS.

---

### Task 7: Wire CLI to Real Components

**Files:**
- Modify: `src/coding_agent/cli.py`
- Test: `tests/test_cli_run.py`

- [ ] **Step 1: Write CLI missing key test**

Write `tests/test_cli_run.py`:

```python
from typer.testing import CliRunner

from coding_agent.cli import app


def test_run_reports_missing_api_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runner = CliRunner()
    runner.invoke(app, ["init"])

    result = runner.invoke(app, ["run", "hello"])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output
```

- [ ] **Step 2: Wire `run` and `chat`**

Replace `src/coding_agent/cli.py` with:

```python
from pathlib import Path

import typer
from rich.console import Console

from coding_agent.agent import AgentLoop
from coding_agent.config import ConfigError, load_config
from coding_agent.llm import LLMError, OpenAICompatibleClient
from coding_agent.policy import CommandPolicy
from coding_agent.sandbox import WorkspaceSandbox
from coding_agent.session import SessionStore
from coding_agent.shell import ShellRunner
from coding_agent.tools import create_default_tools

app = typer.Typer(help="AI coding assistant CLI")
console = Console()


@app.command()
def init(path: Path = typer.Option(Path("."), "--path", "-p", help="Project root")) -> None:
    """Create a project-local coding-agent config."""
    config_dir = path / ".coding-agent"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"
    if config_path.exists():
        console.print(f"[yellow]Config already exists:[/] {config_path}")
        return
    config_path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    console.print(f"[green]Created config:[/] {config_path}")


@app.command()
def run(task: str) -> None:
    """Run one coding task and exit."""
    try:
        result, session_path = _run_task(task, [])
    except (ConfigError, LLMError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc
    console.print(result.final_answer)
    console.print(f"[dim]Session saved:[/] {session_path}")


@app.command()
def chat() -> None:
    """Start an interactive coding session."""
    messages: list[dict] = []
    console.print("[green]coding-agent chat[/] Type /exit to quit, /clear to reset.")
    while True:
        user_input = typer.prompt(">")
        if user_input.strip() == "/exit":
            return
        if user_input.strip() == "/clear":
            messages = []
            console.print("[dim]Context cleared.[/]")
            continue
        if user_input.strip() == "/status":
            console.print(f"[dim]Messages in context:[/] {len(messages)}")
            continue
        try:
            result, session_path = _run_task(user_input, messages)
        except (ConfigError, LLMError) as exc:
            console.print(f"[red]Error:[/] {exc}")
            continue
        messages = result.messages
        console.print(result.final_answer)
        console.print(f"[dim]Session saved:[/] {session_path}")


def _run_task(task: str, prior_messages: list[dict]) -> tuple[object, Path]:
    project_root = Path.cwd()
    config = load_config(project_root)
    sandbox = WorkspaceSandbox(config.workspace.root)
    policy = CommandPolicy(config.commands.allow, config.commands.deny)
    shell = ShellRunner(policy, config.workspace.root)
    tools = create_default_tools(sandbox, shell)
    client = OpenAICompatibleClient(config.model.base_url, config.model.api_key, config.model.model)
    loop = AgentLoop(client, tools, config.agent.max_steps)
    result = loop.run(task, prior_messages)
    session_path = SessionStore(project_root).save(result.messages)
    return result, session_path


DEFAULT_CONFIG = """[model]
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
"""
```

- [ ] **Step 3: Run CLI tests**

Run: `python -m pytest tests/test_cli_init.py tests/test_cli_run.py -v`

Expected: PASS.

---

### Task 8: Final Verification and Documentation

**Files:**
- Create: `README.md`
- Modify: `docs/plans/2026-05-13-python-coding-cli-design.md` only if implementation diverged.

- [ ] **Step 1: Write README**

Write `README.md`:

```markdown
# coding-agent

Python AI coding assistant CLI.

## Install for development

```bash
python -m pip install -e ".[dev]"
```

## Initialize a project

```bash
coding-agent init
```

Set the configured API key environment variable:

```bash
export OPENAI_API_KEY="..."
```

On PowerShell:

```powershell
$env:OPENAI_API_KEY = "..."
```

## Run one task

```bash
coding-agent run "inspect the project and summarize it"
```

## Chat mode

```bash
coding-agent chat
```

Available commands:

- `/exit`
- `/clear`
- `/status`

## Safety model

File access is limited to the configured workspace root, which defaults to the current project. Shell commands must match the configured allow list and must not match the deny list. Deny rules take precedence.
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest -v`

Expected: PASS.

- [ ] **Step 3: Run import smoke test**

Run: `python -c "from coding_agent.cli import app; print(app.info.help)"`

Expected output includes: `AI coding assistant CLI`.

- [ ] **Step 4: Run CLI init manually**

Run in a temporary directory:

```bash
coding-agent init
```

Expected: `.coding-agent/config.toml` is created.

- [ ] **Step 5: Commit if git exists**

Run only in a git repository:

```bash
git add README.md docs/plans docs/superpowers/plans src tests pyproject.toml
git commit -m "feat: implement python coding agent mvp"
```

---

## Self-Review

Spec coverage:

- OpenAI-compatible model access: Task 6.
- `chat` and `run`: Task 7.
- `init` and config file: Task 1 and Task 2.
- Project-only file access: Task 3 and Task 5.
- Configurable allow/deny shell policy: Task 4.
- Tool registry and built-in tools: Task 5.
- Session logging: Task 6 and Task 7.
- Tests for safety and orchestration: Tasks 2 through 7.

Known implementation note:

- `apply_patch` is exposed in the registry but returns an explicit unsupported result in this plan. A follow-up task should implement patch parsing safely before relying on it for real edits.
