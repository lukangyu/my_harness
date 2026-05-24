import json
import os
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

from coding_agent.application import Application
from coding_agent.config import ConfigError, load_config
from coding_agent.interrupts import TaskInterrupted
from coding_agent.llm import LLMError
from coding_agent.run_result import RunTaskResult
from coding_agent.runtime_events import RuntimeEvent
from coding_agent.session import ConversationStore, LegacySessionError, SessionRef

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
        task_result = _run_task(task, None, mode="run")
    except TaskInterrupted as exc:
        console.print(f"[yellow]Task interrupted:[/] {exc or 'user interrupted'}")
        raise typer.Exit(130) from exc
    except (ConfigError, LLMError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    _print_task_result(task_result)


@app.command()
def chat(
    resume: Optional[Path] = typer.Option(None, "--resume", help="Resume a conversation or session directory"),
    resume_latest: bool = typer.Option(False, "--resume-latest", help="Resume the latest conversation"),
) -> None:
    """Start an interactive coding session."""
    try:
        session_ref, messages = _load_chat_session(resume, resume_latest)
    except (OSError, ValueError, LegacySessionError, json.JSONDecodeError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print("Chat mode. Type /exit to quit.", style="dim")

    while True:
        task = typer.prompt("coding-agent")
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
            task_result = _invoke_run_task(command, messages, "chat", session_ref)
        except TaskInterrupted as exc:
            console.print(f"[yellow]Task interrupted:[/] {exc or 'user interrupted'}")
            continue
        except (ConfigError, LLMError) as exc:
            console.print(f"[red]Error:[/] {exc}")
            continue

        result = task_result.result
        messages[:] = result.conversation_messages
        session_parent = Path(task_result.session_path).parent
        if (session_parent / "session.json").exists():
            session_ref = ConversationStore(Path.cwd()).resume(session_parent)
        _print_task_result(task_result)


def _load_chat_session(resume: Optional[Path], resume_latest: bool) -> tuple[SessionRef | None, list[dict[str, Any]]]:
    if resume is not None and resume_latest:
        raise ValueError("--resume and --resume-latest cannot be used together")
    store = ConversationStore(Path.cwd())
    if resume is not None:
        session = store.resume(resume)
        return session, store.load_session_messages(session)
    if resume_latest:
        session = store.latest()
        if session is None:
            return None, []
        return session, store.load_session_messages(session)
    return None, []


def _run_task(
    task: str,
    prior_messages: list[dict[str, Any]] | None,
    mode: str,
    session_ref: SessionRef | None = None,
) -> RunTaskResult:
    config = load_config(Path.cwd())
    renderer_state = _new_renderer_state()
    application = Application(
        config,
        on_reasoning_delta=_make_reasoning_printer(renderer_state),
        on_tool_call=_make_tool_printer(renderer_state),
        on_runtime_event=_make_runtime_event_printer(renderer_state),
        command_approval=_make_command_approval(),
    )
    if not hasattr(config, "project_root"):
        return application.run_task(task, prior_messages, mode)
    conversation_store = ConversationStore(config.project_root)
    return application.run_task(task, prior_messages, mode, session_ref=session_ref, conversation_store=conversation_store)


def _invoke_run_task(
    task: str,
    prior_messages: list[dict[str, Any]],
    mode: str,
    session_ref: SessionRef | None,
) -> RunTaskResult:
    try:
        return _run_task(task, prior_messages, mode, session_ref=session_ref)
    except TypeError as exc:
        if "session_ref" not in str(exc):
            raise
        return _run_task(task, prior_messages, mode)


def _print_task_result(task_result: RunTaskResult) -> None:
    reasoning = _latest_reasoning(task_result.result.conversation_messages)
    if reasoning:
        _print_section("thinking", style="dim")
        console.print(reasoning, style="dim")
    final_answer = task_result.result.final_answer
    if final_answer:
        _print_section("answer", style="bold")
        console.print(Markdown(final_answer))
    else:
        console.print("")

    _print_section("status", style="dim")
    if task_result.show_cache_stats and task_result.result.usage:
        ratio = task_result.result.usage.cache_hit_ratio
        if ratio is not None:
            console.print(f"  Cache: {ratio:.0%} cached input tokens", style="dim")
    console.print(f"  Session: {task_result.session_path}", style="dim", markup=False)
    if task_result.conversation_path is not None:
        console.print(f"  Conversation: {task_result.conversation_path.parent}", style="dim", markup=False)
    if task_result.run_dir is not None:
        console.print(f"  Run: {task_result.run_dir}", style="dim", markup=False)


def _new_renderer_state() -> dict[str, bool]:
    return {"reasoning_active": False, "tools_started": False}


def _make_reasoning_printer(state: dict[str, bool] | None = None) -> Any:
    if state is None:
        state = _new_renderer_state()

    def on_reasoning_delta(text: str) -> None:
        if not text:
            return
        if not state["reasoning_active"]:
            if state["tools_started"]:
                console.print()
            _print_section("thinking", style="dim")
            state["reasoning_active"] = True
        console.out(text, end="")

    return on_reasoning_delta


def _make_tool_printer(state: dict[str, bool] | None = None) -> Any:
    if state is None:
        state = _new_renderer_state()

    def on_tool_call(tool_call: dict[str, Any]) -> None:
        if not state["tools_started"]:
            console.print()
            _print_section("tools", style="cyan")
            state["tools_started"] = True
        state["reasoning_active"] = False
        console.print(f"  -> {_format_tool_call(tool_call)}", style="cyan", markup=False)

    return on_tool_call


def _make_runtime_event_printer(state: dict[str, bool] | None = None) -> Any:
    if state is None:
        state = _new_renderer_state()

    def on_runtime_event(event: RuntimeEvent) -> None:
        if event.type == "tool.result":
            _print_tool_result_event(event, state)
            return
        if event.type != "context.built":
            return
        if state["reasoning_active"]:
            console.print()
            state["reasoning_active"] = False
        metadata = event.metadata
        injected = []
        if metadata.get("memory_anchor"):
            injected.append("memory")
        if metadata.get("handoff_memo"):
            injected.append("handoff")
        injected_text = ", ".join(injected) if injected else "none"
        _print_section("context", style="green")
        console.print(
            (
                f"  prompt: injected={injected_text}; "
                f"recent={metadata.get('recent_messages', 0)}; "
                f"tools={metadata.get('tool_count', 0)}"
            ),
            style="green",
            markup=False,
        )

    return on_runtime_event


def _make_command_approval(read_key: Any | None = None) -> Any:
    if read_key is None:
        read_key = _read_approval_key

    def approve(command: str, reason: str) -> bool:
        console.print(
            (
                f"Allow shell command? {command}\n"
                f"Reason: {reason}\n"
                "Press y to allow, n to reject, Esc to interrupt current task."
            ),
            style="yellow",
            markup=False,
        )
        while True:
            key = read_key()
            normalized = key.lower()
            if normalized == "y":
                console.print("  command approved", style="yellow")
                return True
            if normalized == "n":
                console.print("  command rejected", style="yellow")
                return False
            if key == "\x1b":
                raise TaskInterrupted("user interrupted during shell approval")

    return approve


def _read_approval_key() -> str:
    if os.name == "nt":
        import msvcrt

        return msvcrt.getwch()
    import sys
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _print_tool_result_event(event: RuntimeEvent, state: dict[str, bool]) -> None:
    if state["reasoning_active"]:
        console.print()
        state["reasoning_active"] = False
    if not state["tools_started"]:
        _print_section("tools", style="cyan")
        state["tools_started"] = True
    metadata = event.metadata
    tool = metadata.get("tool") or "unknown"
    if metadata.get("ok") is False:
        console.print(f"  <- {tool} failed: {metadata.get('error') or 'unknown error'}", style="red", markup=False)
        return
    details: list[str] = []
    if metadata.get("path"):
        details.append(f"path={metadata['path']}")
    if "exit_code" in metadata:
        details.append(f"exit_code={metadata['exit_code']}")
    if metadata.get("timed_out"):
        details.append("timed_out=true")
    for key in ("changed_files_count", "matches_count", "files_count"):
        if key in metadata:
            details.append(f"{key}={metadata[key]}")
    suffix = f" ({', '.join(details)})" if details else ""
    console.print(f"  <- {tool} ok{suffix}", style="cyan", markup=False)


def _latest_reasoning(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:
            return reasoning
    return ""


def _format_tool_call(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return "unknown()"
    name = function.get("name")
    if not isinstance(name, str) or not name:
        name = "unknown"
    raw_arguments = function.get("arguments")
    if not isinstance(raw_arguments, str) or not raw_arguments:
        return f"{name}()"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return f"{name}({raw_arguments})"
    if not isinstance(arguments, dict) or not arguments:
        return f"{name}()"
    rendered_arguments = ", ".join(
        f"{key}={value!r}" for key, value in arguments.items()
    )
    return f"{name}({rendered_arguments})"


def _print_section(label: str, *, style: str) -> None:
    console.print(label, style=style)


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
  "git diff",
  "git log"
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
compact_threshold_ratio = 0.8
protected_recent_turns = 4
protected_tool_results = 6
handoff_max_chars = 6000
scratchpad_max_chars = 4000
"""
