from typer.testing import CliRunner

import coding_agent.cli as cli_module
from coding_agent.cli import app
from coding_agent.config import ConfigError


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

    def fail_task(task, prior_messages):
        raise ConfigError("missing config")

    monkeypatch.setattr(cli_module, "_run_task", fail_task)

    result = runner.invoke(app, ["chat"], input="hello\n/status\n/exit\n")

    assert result.exit_code == 0
    assert "missing config" in result.output
    assert "Messages: 0" in result.output
