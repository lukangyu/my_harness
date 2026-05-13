from typer.testing import CliRunner

from coding_agent.cli import app


def test_run_without_api_key_prints_error_and_exits(tmp_path, monkeypatch):
    runner = CliRunner()
    init_result = runner.invoke(app, ["init", "--path", str(tmp_path)])
    assert init_result.exit_code == 0
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["run", "summarize the project"])

    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output
