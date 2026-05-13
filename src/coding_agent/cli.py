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
