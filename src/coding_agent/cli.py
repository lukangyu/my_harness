import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

from coding_agent.agent import AgentLoop, AgentResult
from coding_agent.config import ConfigError, load_config
from coding_agent.context import WorkspaceContextOptions
from coding_agent.llm import LLMError, OpenAICompatibleClient
from coding_agent.memory import MemoryStore
from coding_agent.policy import CommandPolicy
from coding_agent.sandbox import WorkspaceSandbox
from coding_agent.session import SessionStore
from coding_agent.shell import ShellRunner
from coding_agent.telemetry import TelemetryLogger
from coding_agent.tools import create_default_tools

app = typer.Typer(help="AI coding assistant CLI")
console = Console()


@dataclass(frozen=True)
class RunTaskResult:
    result: AgentResult
    session_path: Path
    show_cache_stats: bool


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
    except (ConfigError, LLMError) as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(1) from exc

    _print_task_result(task_result)


@app.command()
def chat(
    resume: Optional[Path] = typer.Option(None, "--resume", help="Load messages from a session JSON file"),
    resume_latest: bool = typer.Option(False, "--resume-latest", help="Load the latest saved session"),
) -> None:
    """Start an interactive coding session."""
    try:
        messages = _load_chat_messages(resume, resume_latest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
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
            task_result = _run_task(command, messages, mode="chat")
        except (ConfigError, LLMError) as exc:
            console.print(f"[red]Error:[/] {exc}")
            continue

        result = task_result.result
        messages[:] = result.conversation_messages
        _print_task_result(task_result)


def _load_chat_messages(resume: Optional[Path], resume_latest: bool) -> list[dict[str, Any]]:
    if resume is not None and resume_latest:
        raise ValueError("--resume and --resume-latest cannot be used together")
    if resume is not None:
        return SessionStore.load(resume)
    if resume_latest:
        return SessionStore(Path.cwd()).load_latest() or []
    return []


def _run_task(
    task: str,
    prior_messages: list[dict[str, Any]] | None,
    mode: str,
) -> RunTaskResult:
    config = load_config(Path.cwd())
    telemetry = TelemetryLogger(
        config.project_root / ".coding-agent" / "logs",
        workspace_root=config.workspace.root,
    )
    memory_store = MemoryStore(config.project_root)
    telemetry.event(
        "cli.task.start",
        "CLI 收到任务并开始执行",
        function="_run_task",
        phase="cli",
        metadata={"mode": mode, "task_length": len(task), "has_prior_messages": bool(prior_messages)},
    )
    telemetry.workspace_snapshot(
        message_zh="任务开始前的 workspace 文件树和目录树快照",
        function="_run_task",
        phase="workspace_before_task",
        root=config.workspace.root,
    )
    sandbox = WorkspaceSandbox(config.workspace.root)
    policy = CommandPolicy(allow=config.commands.allow, deny=config.commands.deny)
    shell = ShellRunner(policy=policy, cwd=sandbox.root)
    tools = create_default_tools(sandbox, shell, telemetry=telemetry, memory_store=memory_store)
    renderer_state = _new_renderer_state()
    client = OpenAICompatibleClient(
        base_url=config.model.base_url,
        api_key=config.model.api_key,
        model=config.model.model,
        stream=config.agent.stream,
        on_reasoning_delta=_make_reasoning_printer(renderer_state),
        debug_dir=config.project_root / ".coding-agent" / "debug",
        telemetry=telemetry,
    )
    context_options = WorkspaceContextOptions(
        doc_max_chars=config.context.doc_max_chars,
        tree_max_entries=config.context.tree_max_entries,
        include_project_docs=config.context.include_project_docs,
        include_file_tree=config.context.include_file_tree,
        include_git_status=config.context.include_git_status,
        include_recent_commits=config.context.include_recent_commits,
        max_input_tokens=config.context.max_input_tokens,
        compact_threshold_ratio=config.context.compact_threshold_ratio,
        protected_recent_turns=config.context.protected_recent_turns,
        protected_tool_results=config.context.protected_tool_results,
        handoff_max_chars=config.context.handoff_max_chars,
        scratchpad_max_chars=config.context.scratchpad_max_chars,
        file_summaries_max_count=config.context.file_summaries_max_count,
        file_summaries_max_chars=config.context.file_summaries_max_chars,
    )
    agent = AgentLoop(
        client=client,
        tools=tools,
        max_steps=config.agent.max_steps,
        cwd=sandbox.root,
        context_options=context_options,
        recent_message_tokens=config.context.recent_message_tokens,
        on_tool_call=_make_tool_printer(renderer_state),
        telemetry=telemetry,
        memory_store=memory_store,
    )
    with telemetry.span(
        "执行完整任务",
        function="_run_task",
        phase="cli",
        metadata={"mode": mode},
    ):
        result = agent.run(task, prior_messages=prior_messages, mode=mode)
    telemetry.workspace_snapshot(
        message_zh="任务结束后的 workspace 文件树和目录树快照",
        function="_run_task",
        phase="workspace_after_task",
        root=config.workspace.root,
    )
    session_path = SessionStore(config.project_root, telemetry=telemetry).save(result.conversation_messages)
    telemetry.event(
        "cli.task.end",
        "CLI 任务执行完成",
        function="_run_task",
        phase="cli",
        metadata={
            "session_path": str(session_path),
            "final_answer_length": len(result.final_answer),
            "reached_max_steps": result.reached_max_steps,
        },
    )
    return RunTaskResult(result, session_path, config.context.show_cache_stats)


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
compact_threshold_ratio = 0.8
protected_recent_turns = 4
protected_tool_results = 6
handoff_max_chars = 6000
scratchpad_max_chars = 4000
file_summaries_max_count = 8
file_summaries_max_chars = 8000
"""
