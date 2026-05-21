from typer.testing import CliRunner
from rich.console import Console

import coding_agent.cli as cli_module
from coding_agent.agent import AgentResult
from coding_agent.cli import RunTaskResult, app
from coding_agent.config import ConfigError
from coding_agent.context import UsageStats
from coding_agent.runtime_events import RuntimeEvent
from coding_agent.session import SessionStore


def test_run_task_delegates_to_application(monkeypatch):
    calls = []
    fake_config = object()

    class FakeApplication:
        def __init__(self, config, *, on_reasoning_delta, on_tool_call, on_runtime_event, command_approval):
            calls.append(
                {
                    "config": config,
                    "has_reasoning": callable(on_reasoning_delta),
                    "has_tool": callable(on_tool_call),
                    "has_runtime_event": callable(on_runtime_event),
                    "has_command_approval": callable(command_approval),
                }
            )

        def run_task(self, task, prior_messages, mode):
            calls.append({"task": task, "prior_messages": prior_messages, "mode": mode})
            return RunTaskResult(
                AgentResult(final_answer="answer", messages=[], conversation_messages=[]),
                "session.json",
                False,
            )

    monkeypatch.setattr(cli_module, "load_config", lambda root: fake_config)
    monkeypatch.setattr(cli_module, "Application", FakeApplication)

    result = cli_module._run_task("summarize", [], "run")

    assert result.result.final_answer == "answer"
    assert calls == [
        {
            "config": fake_config,
            "has_reasoning": True,
            "has_tool": True,
            "has_runtime_event": True,
            "has_command_approval": True,
        },
        {"task": "summarize", "prior_messages": [], "mode": "run"},
    ]


def test_run_without_api_key_prints_error_and_exits(tmp_path, monkeypatch):
    runner = CliRunner()
    init_result = runner.invoke(app, ["init", "--path", str(tmp_path)])
    assert init_result.exit_code == 0
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["run", "summarize the project"])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


def test_chat_reports_task_error_and_continues(monkeypatch):
    runner = CliRunner()
    calls = []

    def fail_task(task, prior_messages, mode):
        calls.append({"task": task, "prior_messages": prior_messages, "mode": mode})
        raise ConfigError("missing config")

    monkeypatch.setattr(cli_module, "_run_task", fail_task)

    result = runner.invoke(app, ["chat"], input="hello\n/status\n/exit\n")

    assert result.exit_code == 0
    assert "missing config" in result.output
    assert "Messages: 0" in result.output
    assert calls == [{"task": "hello", "prior_messages": [], "mode": "chat"}]


def test_chat_reuses_conversation_messages(monkeypatch):
    runner = CliRunner()
    calls = []

    def run_task(task, prior_messages, mode):
        calls.append({"task": task, "prior_messages": list(prior_messages), "mode": mode})
        return RunTaskResult(
            AgentResult(
                final_answer=f"answer to {task}",
                messages=[
                    {"role": "system", "content": "<coding_agent_prefix>"},
                    {"role": "user", "content": "<workspace_context>"},
                    {"role": "user", "content": f"<current_task>{task}</current_task>"},
                    {"role": "assistant", "content": f"answer to {task}"},
                ],
                conversation_messages=[
                    {"role": "user", "content": task},
                    {"role": "assistant", "content": f"answer to {task}"},
                ],
            ),
            "session.json",
            False,
        )

    monkeypatch.setattr(cli_module, "_run_task", run_task)

    result = runner.invoke(app, ["chat"], input="first\n/status\nsecond\n/exit\n")

    assert result.exit_code == 0
    assert "Messages: 2" in result.output
    assert calls == [
        {"task": "first", "prior_messages": [], "mode": "chat"},
        {
            "task": "second",
            "prior_messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer to first"},
            ],
            "mode": "chat",
        },
    ]


def test_chat_resume_path_reports_loaded_message_count(tmp_path, monkeypatch):
    runner = CliRunner()
    calls = []
    session_path = tmp_path / "session.json"
    session_path.write_text(
        '[{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "answer"}]',
        encoding="utf-8",
    )

    def run_task(task, prior_messages, mode):
        calls.append({"task": task, "prior_messages": list(prior_messages), "mode": mode})
        return RunTaskResult(
            AgentResult(
                final_answer="next answer",
                messages=[],
                conversation_messages=[
                    *prior_messages,
                    {"role": "user", "content": task},
                    {"role": "assistant", "content": "next answer"},
                ],
            ),
            "session.json",
            False,
        )

    monkeypatch.setattr(cli_module, "_run_task", run_task)

    result = runner.invoke(app, ["chat", "--resume", str(session_path)], input="/status\nnext\n/exit\n")

    assert result.exit_code == 0
    assert "Messages: 2" in result.output
    assert calls == [
        {
            "task": "next",
            "prior_messages": [
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "answer"},
            ],
            "mode": "chat",
        }
    ]


def test_chat_resume_latest_reports_loaded_message_count(tmp_path, monkeypatch):
    runner = CliRunner()
    store = SessionStore(tmp_path)
    store.sessions_dir.mkdir(parents=True)
    latest = store.sessions_dir / "20260513-120000-000002.json"
    latest.write_text('[{"role": "user", "content": "latest"}]', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "_run_task", lambda task, prior_messages, mode: None)

    result = runner.invoke(app, ["chat", "--resume-latest"], input="/status\n/exit\n")

    assert result.exit_code == 0
    assert "Messages: 1" in result.output


def test_chat_invalid_resume_exits_with_error(tmp_path):
    runner = CliRunner()
    session_path = tmp_path / "session.json"
    session_path.write_text('{"role": "user"}', encoding="utf-8")

    result = runner.invoke(app, ["chat", "--resume", str(session_path)], input="/status\n")

    assert result.exit_code == 1
    assert "Error:" in result.output


def test_chat_resume_and_resume_latest_conflict_exits_with_error(tmp_path):
    runner = CliRunner()
    session_path = tmp_path / "session.json"
    session_path.write_text('[{"role": "user", "content": "earlier"}]', encoding="utf-8")

    result = runner.invoke(
        app,
        ["chat", "--resume", str(session_path), "--resume-latest"],
        input="/status\n",
    )

    assert result.exit_code == 1
    assert "--resume and --resume-latest cannot be used together" in result.output


def test_run_prints_cache_stats_when_enabled_and_usage_has_ratio(monkeypatch):
    runner = CliRunner()

    def run_task(task, prior_messages, mode):
        return RunTaskResult(
            AgentResult(
                final_answer="answer",
                messages=[],
                conversation_messages=[],
                usage=UsageStats(input_tokens=100, output_tokens=10, cached_tokens=85),
            ),
            "session.json",
            True,
        )

    monkeypatch.setattr(cli_module, "_run_task", run_task)

    result = runner.invoke(app, ["run", "summarize"])

    assert result.exit_code == 0
    assert "answer" in result.output
    assert "Cache: 85% cached input tokens" in result.output


def test_run_prints_reasoning_separately_from_final_answer(monkeypatch):
    runner = CliRunner()

    def run_task(task, prior_messages, mode):
        return RunTaskResult(
            AgentResult(
                final_answer="final answer",
                messages=[],
                conversation_messages=[
                    {
                        "role": "assistant",
                        "content": "final answer",
                        "reasoning_content": "thinking text",
                    }
                ],
                usage=UsageStats(input_tokens=100, output_tokens=10, cached_tokens=0),
            ),
            "session.json",
            False,
        )

    monkeypatch.setattr(cli_module, "_run_task", run_task)

    result = runner.invoke(app, ["run", "summarize"])

    assert result.exit_code == 0
    assert "thinking" in result.output
    assert "thinking text" in result.output
    assert "answer" in result.output
    assert "final answer" in result.output


def test_format_tool_call_renders_compact_summary():
    rendered = cli_module._format_tool_call(
        {
            "function": {
                "name": "read_file",
                "arguments": '{"path":"src/coding_agent/agent.py"}',
            }
        }
    )

    assert rendered == "read_file(path='src/coding_agent/agent.py')"


def test_tool_printer_renders_codex_style_event(monkeypatch):
    buffer = Console(record=True, width=100)
    monkeypatch.setattr(cli_module, "console", buffer)
    printer = cli_module._make_tool_printer()

    printer(
        {
            "function": {
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            }
        }
    )

    output = buffer.export_text()
    assert "tools" in output
    assert "-> read_file(path='README.md')" in output


def test_runtime_event_printer_renders_prompt_context(monkeypatch):
    buffer = Console(record=True, width=100)
    monkeypatch.setattr(cli_module, "console", buffer)
    printer = cli_module._make_runtime_event_printer()

    printer(
        RuntimeEvent(
            type="context.built",
            message="上下文已组装",
            metadata={
                "memory_anchor": True,
                "handoff_memo": True,
                "file_summaries": True,
                "recent_messages": 3,
                "tool_count": 6,
            },
        )
    )

    output = buffer.export_text()
    assert "context" in output
    assert "memory" in output
    assert "handoff" in output
    assert "file summaries" in output
    assert "recent=3" in output


def test_runtime_event_printer_renders_failed_tool_result(monkeypatch):
    buffer = Console(record=True, width=100)
    monkeypatch.setattr(cli_module, "console", buffer)
    printer = cli_module._make_runtime_event_printer()

    printer(
        RuntimeEvent(
            type="tool.result",
            message="工具 run_shell 执行完成",
            metadata={
                "tool": "run_shell",
                "ok": False,
                "error": "Command not in allow list",
                "exit_code": None,
                "timed_out": False,
            },
        )
    )

    output = buffer.export_text()
    assert "tools" in output
    assert "<- run_shell failed: Command not in allow list" in output


def test_command_approval_asks_user(monkeypatch):
    prompts = []

    def fake_confirm(message, default):
        prompts.append({"message": message, "default": default})
        return True

    monkeypatch.setattr(cli_module.typer, "confirm", fake_confirm)
    approval = cli_module._make_command_approval()

    assert approval("git log --oneline -n 8", "Command not in allow list") is True
    assert prompts == [
        {
            "message": "Allow shell command? git log --oneline -n 8\nReason: Command not in allow list",
            "default": False,
        }
    ]


def test_print_task_result_renders_markdown(monkeypatch):
    buffer = Console(record=True, width=100)
    monkeypatch.setattr(cli_module, "console", buffer)
    task_result = RunTaskResult(
        AgentResult(
            final_answer="**bold**\n\n- item",
            messages=[],
            conversation_messages=[],
        ),
        "session.json",
        False,
    )

    cli_module._print_task_result(task_result)

    output = buffer.export_text()
    assert "answer" in output
    assert "**bold**" not in output
    assert "bold" in output
    assert "item" in output
    assert "status" in output


def test_run_omits_cache_stats_when_disabled(monkeypatch):
    runner = CliRunner()

    def run_task(task, prior_messages, mode):
        return RunTaskResult(
            AgentResult(
                final_answer="answer",
                messages=[],
                conversation_messages=[],
                usage=UsageStats(input_tokens=100, output_tokens=10, cached_tokens=85),
            ),
            "session.json",
            False,
        )

    monkeypatch.setattr(cli_module, "_run_task", run_task)

    result = runner.invoke(app, ["run", "summarize"])

    assert result.exit_code == 0
    assert "Cache:" not in result.output


def test_chat_resume_preserves_reasoning_content(tmp_path):
    session_path = tmp_path / "session.json"
    session_path.write_text(
        '[{"role": "assistant", "content": "answer", "reasoning_content": "thinking"}]',
        encoding="utf-8",
    )

    records = SessionStore.load(session_path)

    assert records == [
        {"role": "assistant", "content": "answer", "reasoning_content": "thinking"}
    ]
