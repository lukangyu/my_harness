from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from coding_agent.agent import AgentLoop, AgentResult
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
        result, session_path = _run_task(task, None)
    except (ConfigError, LLMError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print(result.final_answer)
    console.print(f"Session: {session_path}")


@app.command()
def chat() -> None:
    """Start an interactive coding session."""
    messages: list[dict[str, Any]] = []
    console.print("Chat mode. Type /exit to quit.")

    while True:
        task = typer.prompt("> ")
        command = task.strip()
        if command == "/exit":
            return
        if command == "/clear":
            messages.clear()
            console.print("Cleared conversation.")
            continue
        if command == "/status":
            console.print(f"Messages: {len(messages)}")
            continue
        if not command:
            continue

        try:
            result, session_path = _run_task(command, messages)
        except (ConfigError, LLMError) as exc:
            console.print(f"[red]Error:[/] {exc}")
            continue

        messages[:] = result.messages
        console.print(result.final_answer)
        console.print(f"Session: {session_path}")


def _run_task(task: str, prior_messages: list[dict[str, Any]] | None) -> tuple[AgentResult, Path]:
    config = load_config(Path.cwd())
    sandbox = WorkspaceSandbox(config.workspace.root)
    policy = CommandPolicy(allow=config.commands.allow, deny=config.commands.deny)
    shell = ShellRunner(policy=policy, cwd=sandbox.root)
    tools = create_default_tools(sandbox, shell)
    client = OpenAICompatibleClient(
        base_url=config.model.base_url,
        api_key=config.model.api_key,
        model=config.model.model,
    )
    agent = AgentLoop(client=client, tools=tools, max_steps=config.agent.max_steps)
    result = agent.run(task, prior_messages=prior_messages)
    session_path = SessionStore(config.project_root).save(result.messages)
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
