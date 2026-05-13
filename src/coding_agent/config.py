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
class AppConfig:
    project_root: Path
    model: ModelConfig
    agent: AgentConfig
    workspace: WorkspaceConfig
    commands: CommandConfig


def load_config(project_root: Path) -> AppConfig:
    project_root = project_root.resolve()
    config_path = project_root / ".coding-agent" / "config.toml"
    if not config_path.exists():
        raise ConfigError("Missing .coding-agent/config.toml. Run coding-agent init first.")

    with config_path.open("rb") as config_file:
        data = tomllib.load(config_file)

    model_data = _required_table(data, "model")
    agent_data = _required_table(data, "agent")
    workspace_data = _required_table(data, "workspace")
    command_data = _required_table(data, "commands")

    api_key_env = str(_required_value(model_data, "api_key_env"))
    try:
        api_key = os.environ[api_key_env]
    except KeyError as exc:
        raise ConfigError(f"Missing required environment variable: {api_key_env}") from exc

    workspace_root = Path(str(_required_value(workspace_data, "root")))
    if not workspace_root.is_absolute():
        workspace_root = project_root / workspace_root

    return AppConfig(
        project_root=project_root,
        model=ModelConfig(
            base_url=str(_required_value(model_data, "base_url")).rstrip("/"),
            api_key_env=api_key_env,
            api_key=api_key,
            model=str(_required_value(model_data, "model")),
        ),
        agent=AgentConfig(
            max_steps=int(_required_value(agent_data, "max_steps")),
            stream=bool(_required_value(agent_data, "stream")),
        ),
        workspace=WorkspaceConfig(root=workspace_root.resolve()),
        commands=CommandConfig(
            allow=[str(item) for item in _required_value(command_data, "allow")],
            deny=[str(item) for item in _required_value(command_data, "deny")],
        ),
    )


def _required_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = _required_value(data, key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing required configuration key: {key}")
    return value


def _required_value(data: dict[str, Any], key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"Missing required configuration key: {key}")
        current = current[part]
    return current
