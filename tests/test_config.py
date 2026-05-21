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


def test_load_config_uses_context_defaults_when_section_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
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

    config = load_config(tmp_path)

    assert config.context.max_input_tokens == 24000
    assert config.context.reserved_output_tokens == 4000
    assert config.context.recent_message_tokens == 12000
    assert config.context.project_context_tokens == 4000
    assert config.context.doc_max_chars == 1200
    assert config.context.tree_max_entries == 200
    assert config.context.include_project_docs is True
    assert config.context.include_file_tree is True
    assert config.context.include_git_status is True
    assert config.context.include_recent_commits is True
    assert config.context.restore_last_session is False
    assert config.context.show_cache_stats is True
    assert config.context.compact_threshold_ratio == 0.8
    assert config.context.protected_recent_turns == 4
    assert config.context.protected_tool_results == 6
    assert config.context.handoff_max_chars == 6000
    assert config.context.scratchpad_max_chars == 4000
    assert config.context.file_summaries_max_count == 8
    assert config.context.file_summaries_max_chars == 8000


def test_load_config_reads_context_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
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

[context]
max_input_tokens = 1000
reserved_output_tokens = 100
recent_message_tokens = 500
project_context_tokens = 200
doc_max_chars = 300
tree_max_entries = 20
include_project_docs = false
include_file_tree = false
include_git_status = false
include_recent_commits = false
restore_last_session = true
show_cache_stats = false
compact_threshold_ratio = 0.7
protected_recent_turns = 2
protected_tool_results = 3
handoff_max_chars = 1200
scratchpad_max_chars = 800
file_summaries_max_count = 5
file_summaries_max_chars = 1500
""",
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config.context.max_input_tokens == 1000
    assert config.context.reserved_output_tokens == 100
    assert config.context.recent_message_tokens == 500
    assert config.context.project_context_tokens == 200
    assert config.context.doc_max_chars == 300
    assert config.context.tree_max_entries == 20
    assert config.context.include_project_docs is False
    assert config.context.include_file_tree is False
    assert config.context.include_git_status is False
    assert config.context.include_recent_commits is False
    assert config.context.restore_last_session is True
    assert config.context.show_cache_stats is False
    assert config.context.compact_threshold_ratio == 0.7
    assert config.context.protected_recent_turns == 2
    assert config.context.protected_tool_results == 3
    assert config.context.handoff_max_chars == 1200
    assert config.context.scratchpad_max_chars == 800
    assert config.context.file_summaries_max_count == 5
    assert config.context.file_summaries_max_chars == 1500


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


def test_load_config_rejects_non_table_context(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_dir = tmp_path / ".coding-agent"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        """
context = "not-a-table"

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

    with pytest.raises(ConfigError, match="context.*table"):
        load_config(tmp_path)


@pytest.mark.parametrize(
    ("context_line", "missing_key"),
    [
        ("max_input_tokens = true", "context.max_input_tokens"),
        ('include_file_tree = "true"', "context.include_file_tree"),
    ],
)
def test_load_config_rejects_invalid_context_value_types(
    tmp_path, monkeypatch, context_line, missing_key
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config_dir = tmp_path / ".coding-agent"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        f"""
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

[context]
{context_line}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=missing_key):
        load_config(tmp_path)
