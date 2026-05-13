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


@pytest.mark.parametrize(
    ("replacement", "missing_key"),
    [
        ("stream = \"false\"", "agent.stream"),
        ("max_steps = true", "agent.max_steps"),
        ("allow = \"pytest\"", "commands.allow"),
    ],
)
def test_load_config_rejects_invalid_value_types(tmp_path, monkeypatch, replacement, missing_key):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_dir = tmp_path / ".coding-agent"
    config_dir.mkdir()
    config_text = """
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
allow = ["pytest"]
deny = []
"""
    if missing_key == "agent.stream":
        config_text = config_text.replace("stream = true", replacement)
    elif missing_key == "agent.max_steps":
        config_text = config_text.replace("max_steps = 20", replacement)
    else:
        config_text = config_text.replace("allow = [\"pytest\"]", replacement)
    (config_dir / "config.toml").write_text(config_text, encoding="utf-8")

    with pytest.raises(ConfigError, match=missing_key):
        load_config(tmp_path)


def test_load_config_wraps_malformed_toml(tmp_path):
    config_dir = tmp_path / ".coding-agent"
    config_dir.mkdir()
    config_path = config_dir / "config.toml"
    config_path.write_text("[model\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="config.toml"):
        load_config(tmp_path)


def test_load_config_reports_non_table_values(tmp_path):
    config_dir = tmp_path / ".coding-agent"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        """
model = "not-a-table"

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

    with pytest.raises(ConfigError, match="model.*table"):
        load_config(tmp_path)
