from typer.testing import CliRunner

import coding_agent.cli as cli_module
from coding_agent.agent import AgentResult
from coding_agent.cli import app
from coding_agent.config import ConfigError
from coding_agent.session import SessionStore


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
        return (
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
        return (
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
