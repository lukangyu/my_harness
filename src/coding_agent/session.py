from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from coding_agent.telemetry.logger import TelemetryLogger

_ALLOWED_ROLES = {"user", "assistant", "tool"}
_GENERATED_PROMPT_MARKERS = (
    "<coding_agent_prefix",
    "<workspace_context>",
    "<current_task>",
)


class SessionStore:
    def __init__(self, project_root: Path | str, telemetry: TelemetryLogger | None = None) -> None:
        self.project_root = Path(project_root)
        self.sessions_dir = self.project_root / ".coding-agent" / "sessions"
        self.telemetry = telemetry

    def save(self, records: list[dict[str, Any]]) -> Path:
        if self.telemetry is not None:
            self.telemetry.event(
                "session.save.start",
                "开始保存会话记录",
                function="SessionStore.save",
                phase="session",
                metadata={"record_count": len(records)},
            )
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        path = self.sessions_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.telemetry is not None:
            self.telemetry.event(
                "session.save.end",
                "会话记录已保存",
                function="SessionStore.save",
                phase="session",
                metadata={"path": str(path), "record_count": len(records)},
            )
        return path

    @staticmethod
    def load(path: Path | str) -> list[dict[str, Any]]:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("Session JSON root must be a list")
        return [_validate_session_record(record, index) for index, record in enumerate(records)]

    def load_with_telemetry(self, path: Path | str) -> list[dict[str, Any]]:
        if self.telemetry is not None:
            self.telemetry.event(
                "session.load.start",
                "开始加载会话记录",
                function="SessionStore.load_with_telemetry",
                phase="session",
                metadata={"path": str(path)},
            )
        records = self.load(path)
        if self.telemetry is not None:
            self.telemetry.event(
                "session.load.end",
                "会话记录加载完成",
                function="SessionStore.load_with_telemetry",
                phase="session",
                metadata={"path": str(path), "record_count": len(records)},
            )
        return records

    def latest(self) -> Path | None:
        if self.telemetry is not None:
            self.telemetry.event(
                "session.latest.start",
                "开始查找最近的会话记录",
                function="SessionStore.latest",
                phase="session",
            )
        if not self.sessions_dir.exists():
            return None
        sessions = list(self.sessions_dir.glob("*.json"))
        if not sessions:
            return None
        latest = max(sessions, key=lambda path: (path.stat().st_mtime_ns, path.name))
        if self.telemetry is not None:
            self.telemetry.event(
                "session.latest.end",
                "最近的会话记录查找完成",
                function="SessionStore.latest",
                phase="session",
                metadata={"path": str(latest)},
            )
        return latest

    def load_latest(self) -> list[dict[str, Any]] | None:
        path = self.latest()
        if path is None:
            return None
        return self.load_with_telemetry(path)


def _validate_session_record(record: Any, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"Session record {index} must be a dict")

    role = record.get("role")
    if role not in _ALLOWED_ROLES:
        raise ValueError(f"Session record {index} has invalid role")

    if role == "tool":
        _require_string_field(record, index, "tool_call_id")
        _require_string_field(record, index, "name")
        _require_string_field(record, index, "content")

    content = record.get("content")
    if role == "user" and not isinstance(content, str):
        raise ValueError(f"Session record {index} content must be a string")
    if role == "assistant" and content is not None and not isinstance(content, str):
        raise ValueError(f"Session record {index} content must be a string or null")
    reasoning_content = record.get("reasoning_content")
    if role == "assistant" and reasoning_content is not None and not isinstance(reasoning_content, str):
        raise ValueError(f"Session record {index} reasoning_content must be a string or null")
    if isinstance(content, str) and any(marker in content for marker in _GENERATED_PROMPT_MARKERS):
        raise ValueError(f"Session record {index} contains generated prompt block")
    if isinstance(reasoning_content, str) and any(marker in reasoning_content for marker in _GENERATED_PROMPT_MARKERS):
        raise ValueError(f"Session record {index} contains generated prompt block")

    normalized = dict(record)
    validated_tool_calls = None
    if role == "assistant" and "tool_calls" in record:
        tool_calls = record["tool_calls"]
        if not isinstance(tool_calls, list):
            raise ValueError(f"Session record {index} tool_calls must be a list")
        validated_tool_calls = [_validate_tool_call(tool_call, index) for tool_call in tool_calls]
        normalized["tool_calls"] = validated_tool_calls
    if role == "assistant" and content is None and not validated_tool_calls:
        raise ValueError(f"Session record {index} content null requires non-empty tool_calls")
    return normalized


def _require_string_field(record: dict[str, Any], index: int, field: str) -> None:
    if not isinstance(record.get(field), str):
        raise ValueError(f"Session record {index} field {field} must be a string")


def _validate_tool_call(tool_call: Any, index: int) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        raise ValueError(f"Session record {index} tool_calls entries must be dicts")
    if not isinstance(tool_call.get("id"), str):
        raise ValueError(f"Session record {index} tool_calls entries must include string id")
    if tool_call.get("type") != "function":
        raise ValueError(f"Session record {index} tool_calls entries must have function type")
    function = tool_call.get("function")
    if not isinstance(function, dict):
        raise ValueError(f"Session record {index} tool_calls entries must include function dict")
    if not isinstance(function.get("name"), str) or not function["name"]:
        raise ValueError(f"Session record {index} tool_calls entries must include non-empty function name")
    if not isinstance(function.get("arguments"), str):
        raise ValueError(f"Session record {index} tool_calls entries must include string function arguments")
    return dict(tool_call)
