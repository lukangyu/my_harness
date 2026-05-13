from typer.testing import CliRunner

from coding_agent.cli import app


def test_init_creates_project_config(tmp_path):
    runner = CliRunner()

    result = runner.invoke(app, ["init", "--path", str(tmp_path)])

    assert result.exit_code == 0
    config_path = tmp_path / ".coding-agent" / "config.toml"
    assert config_path.exists()
    assert "base_url" in config_path.read_text(encoding="utf-8")
