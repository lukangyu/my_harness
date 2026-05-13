from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    api_key_env: str
    api_key: str
    model: str


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int
    stream: bool


@dataclass(frozen=True)
class WorkspaceConfig:
    root: Path


@dataclass(frozen=True)
class CommandConfig:
    allow: list[str]
    deny: list[str]


@dataclass(frozen=True)
class ContextConfig:
    max_input_tokens: int = 24000
    reserved_output_tokens: int = 4000
    recent_message_tokens: int = 12000
    project_context_tokens: int = 4000
    doc_max_chars: int = 1200
    tree_max_entries: int = 200
    include_project_docs: bool = True
    include_file_tree: bool = True
    include_git_status: bool = True
    include_recent_commits: bool = True
    restore_last_session: bool = False
    show_cache_stats: bool = True


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    model: ModelConfig
    agent: AgentConfig
    workspace: WorkspaceConfig
    commands: CommandConfig
    context: ContextConfig


def load_config(project_root: Path) -> AppConfig:
    project_root = project_root.resolve()
    config_path = project_root / ".coding-agent" / "config.toml"
    if not config_path.exists():
        raise ConfigError("Missing .coding-agent/config.toml. Run coding-agent init first.")

    try:
        with config_path.open("rb") as config_file:
            data = tomllib.load(config_file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Failed to parse {config_path}: {exc}") from exc

    model_data = _required_table(data, "model")
    agent_data = _required_table(data, "agent")
    workspace_data = _required_table(data, "workspace")
    command_data = _required_table(data, "commands")
    context_data = _optional_table(data, "context")

    api_key_env = _required_str(model_data, "api_key_env", "model.api_key_env")
    try:
        api_key = os.environ[api_key_env]
    except KeyError as exc:
        raise ConfigError(f"Missing required environment variable: {api_key_env}") from exc

    workspace_root = Path(_required_str(workspace_data, "root", "workspace.root"))
    if not workspace_root.is_absolute():
        workspace_root = project_root / workspace_root

    context_defaults = ContextConfig()
    context = ContextConfig(
        max_input_tokens=_optional_int(
            context_data,
            "max_input_tokens",
            context_defaults.max_input_tokens,
            "context.max_input_tokens",
        ),
        reserved_output_tokens=_optional_int(
            context_data,
            "reserved_output_tokens",
            context_defaults.reserved_output_tokens,
            "context.reserved_output_tokens",
        ),
        recent_message_tokens=_optional_int(
            context_data,
            "recent_message_tokens",
            context_defaults.recent_message_tokens,
            "context.recent_message_tokens",
        ),
        project_context_tokens=_optional_int(
            context_data,
            "project_context_tokens",
            context_defaults.project_context_tokens,
            "context.project_context_tokens",
        ),
        doc_max_chars=_optional_int(
            context_data,
            "doc_max_chars",
            context_defaults.doc_max_chars,
            "context.doc_max_chars",
        ),
        tree_max_entries=_optional_int(
            context_data,
            "tree_max_entries",
            context_defaults.tree_max_entries,
            "context.tree_max_entries",
        ),
        include_project_docs=_optional_bool(
            context_data,
            "include_project_docs",
            context_defaults.include_project_docs,
            "context.include_project_docs",
        ),
        include_file_tree=_optional_bool(
            context_data,
            "include_file_tree",
            context_defaults.include_file_tree,
            "context.include_file_tree",
        ),
        include_git_status=_optional_bool(
            context_data,
            "include_git_status",
            context_defaults.include_git_status,
            "context.include_git_status",
        ),
        include_recent_commits=_optional_bool(
            context_data,
            "include_recent_commits",
            context_defaults.include_recent_commits,
            "context.include_recent_commits",
        ),
        restore_last_session=_optional_bool(
            context_data,
            "restore_last_session",
            context_defaults.restore_last_session,
            "context.restore_last_session",
        ),
        show_cache_stats=_optional_bool(
            context_data,
            "show_cache_stats",
            context_defaults.show_cache_stats,
            "context.show_cache_stats",
        ),
    )

    return AppConfig(
        project_root=project_root,
        model=ModelConfig(
            base_url=_required_str(model_data, "base_url", "model.base_url").rstrip("/"),
            api_key_env=api_key_env,
            api_key=api_key,
            model=_required_str(model_data, "model", "model.model"),
        ),
        agent=AgentConfig(
            max_steps=_required_int(agent_data, "max_steps", "agent.max_steps"),
            stream=_required_bool(agent_data, "stream", "agent.stream"),
        ),
        workspace=WorkspaceConfig(root=workspace_root.resolve()),
        commands=CommandConfig(
            allow=_required_str_list(command_data, "allow", "commands.allow"),
            deny=_required_str_list(command_data, "deny", "commands.deny"),
        ),
        context=context,
    )


def _required_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required_value(data, key)
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration key {key} must be a table")
    return value


def _optional_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration key {key} must be a table")
    return value


def _required_value(data: dict[str, Any], key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"Missing required configuration key: {key}")
        current = current[part]
    return current


def _required_str(data: dict[str, Any], key: str, display_key: str) -> str:
    value = _required_value(data, key)
    if not isinstance(value, str):
        raise ConfigError(f"Configuration key {display_key} must be a string")
    return value


def _required_int(data: dict[str, Any], key: str, display_key: str) -> int:
    value = _required_value(data, key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"Configuration key {display_key} must be an integer")
    return value


def _optional_int(data: dict[str, Any], key: str, default: int, display_key: str) -> int:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"Configuration key {display_key} must be an integer")
    return value


def _required_bool(data: dict[str, Any], key: str, display_key: str) -> bool:
    value = _required_value(data, key)
    if not isinstance(value, bool):
        raise ConfigError(f"Configuration key {display_key} must be a boolean")
    return value


def _optional_bool(data: dict[str, Any], key: str, default: bool, display_key: str) -> bool:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise ConfigError(f"Configuration key {display_key} must be a boolean")
    return value


def _required_str_list(data: dict[str, Any], key: str, display_key: str) -> list[str]:
    value = _required_value(data, key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(f"Configuration key {display_key} must be a list of strings")
    return value
