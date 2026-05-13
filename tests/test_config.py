import pytest

from coding_agent.config import ConfigError, load_config


def test_load_config_reads_project_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_dir = tmp_path / ".coding-agent"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        """
[model]
base_url = "http://localhost:1234/v1/"
api_key_env = "OPENAI_API_KEY"
model = "local-model"

[agent]
max_steps = 3
stream = false

[workspace]
root = "workspace"

[commands]
allow = ["pytest"]
deny = ["rm"]
""",
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config.project_root == tmp_path.resolve()
    assert config.model.base_url == "http://localhost:1234/v1"
    assert config.model.api_key_env == "OPENAI_API_KEY"
    assert config.model.api_key == "test-key"
    assert config.model.model == "local-model"
    assert config.agent.max_steps == 3
    assert config.agent.stream is False
    assert config.workspace.root == (tmp_path / "workspace").resolve()
    assert config.commands.allow == ["pytest"]
    assert config.commands.deny == ["rm"]


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
