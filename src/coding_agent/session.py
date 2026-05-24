from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
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
LEGACY_SESSION_ERROR = (
    "legacy flat session files are no longer supported; "
    "resume a conversation directory or session directory instead"
)


class LegacySessionError(ValueError):
    pass


@dataclass(frozen=True)
class ConversationRef:
    conversation_id: str
    conversation_dir: Path
    conversation_path: Path
    sessions_dir: Path
    active_session_id: str


@dataclass(frozen=True)
class SessionRef:
    conversation_id: str
    session_id: str
    conversation_dir: Path
    conversation_path: Path
    session_dir: Path
    session_path: Path
    memory_dir: Path
    runs_dir: Path


@dataclass
class SessionRuntime:
    store: "ConversationStore"
    current: SessionRef


class ConversationStore:
    def __init__(self, project_root: Path | str, telemetry: TelemetryLogger | None = None) -> None:
        self.project_root = Path(project_root)
        self.conversations_dir = self.project_root / ".coding-agent" / "conversations"
        self.telemetry = telemetry

    def start_conversation(self) -> SessionRef:
        conversation_id = _new_id()
        session_id = _new_id()
        conversation_dir = self.conversations_dir / conversation_id
        sessions_dir = conversation_dir / "sessions"
        session = self._session_ref(conversation_id, session_id)
        now = _now()
        conversation = {
            "conversation_id": conversation_id,
            "active_session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "sessions": [
                {
                    "session_id": session_id,
                    "status": "active",
                    "previous_session_id": None,
                    "next_session_id": None,
                    "created_at": now,
                    "compacted_at": None,
                }
            ],
        }
        session_payload = _session_payload(
            conversation_id=conversation_id,
            session_id=session_id,
            status="active",
            previous_session_id=None,
            next_session_id=None,
            compacted_at=None,
            messages=[],
            created_at=now,
            updated_at=now,
        )
        session.memory_dir.mkdir(parents=True, exist_ok=True)
        session.runs_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(conversation_dir / "conversation.json", conversation)
        self._atomic_write_json(session.session_path, session_payload)
        return session

    def resume(self, path: Path | str) -> SessionRef:
        candidate = Path(path)
        if candidate.is_file():
            raise LegacySessionError(LEGACY_SESSION_ERROR)
        if not candidate.exists():
            raise FileNotFoundError(f"session path not found: {candidate}")
        if (candidate / "conversation.json").is_file():
            conversation = self._load_conversation(candidate / "conversation.json")
            return self._session_ref(conversation["conversation_id"], conversation["active_session_id"])
        if (candidate / "session.json").is_file():
            session_payload = self._load_session(candidate / "session.json")
            return self._session_ref(session_payload["conversation_id"], session_payload["session_id"])
        raise ValueError("resume path must be a conversation directory or session directory")

    def active_session(self, conversation_id: str) -> SessionRef:
        conversation_dir = self.conversations_dir / conversation_id
        conversation = self._load_conversation(conversation_dir / "conversation.json")
        return self._session_ref(conversation_id, conversation["active_session_id"])

    def latest(self) -> SessionRef | None:
        if not self.conversations_dir.exists():
            return None
        candidates = [path for path in self.conversations_dir.iterdir() if (path / "conversation.json").is_file()]
        if not candidates:
            return None
        latest_dir = max(candidates, key=lambda path: ((path / "conversation.json").stat().st_mtime_ns, path.name))
        return self.resume(latest_dir)

    def save_session_messages(self, session: SessionRef, messages: list[dict[str, Any]]) -> None:
        payload = self._load_session(session.session_path)
        payload["messages"] = [_validate_session_record(record, index) for index, record in enumerate(messages)]
        payload["updated_at"] = _now()
        self._atomic_write_json(session.session_path, payload)

    def load_session_messages(self, session: SessionRef) -> list[dict[str, Any]]:
        payload = self._load_session(session.session_path)
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("session.json messages must be a list")
        return [_validate_session_record(record, index) for index, record in enumerate(messages)]

    def compact_session(
        self,
        session: SessionRef,
        *,
        seed_messages: list[dict[str, Any]],
        summary: str,
    ) -> SessionRef:
        conversation = self._load_conversation(session.conversation_path)
        old_payload = self._load_session(session.session_path)
        new_session_id = _new_id()
        new_session = self._session_ref(session.conversation_id, new_session_id)
        now = _now()
        validated_seed = [_validate_session_record(record, index) for index, record in enumerate(seed_messages)]
        new_payload = _session_payload(
            conversation_id=session.conversation_id,
            session_id=new_session_id,
            status="active",
            previous_session_id=session.session_id,
            next_session_id=None,
            compacted_at=None,
            messages=validated_seed,
            created_at=now,
            updated_at=now,
        )
        new_session.memory_dir.mkdir(parents=True, exist_ok=True)
        new_session.runs_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write_json(new_session.session_path, new_payload)
        (new_session.memory_dir / "handoff.md").write_text(summary, encoding="utf-8")
        old_scratchpad = session.memory_dir / "scratchpad.json"
        new_scratchpad = new_session.memory_dir / "scratchpad.json"
        if old_scratchpad.exists():
            shutil.copyfile(old_scratchpad, new_scratchpad)
        else:
            new_scratchpad.write_text("{}", encoding="utf-8")

        old_payload["status"] = "compacted"
        old_payload["next_session_id"] = new_session_id
        old_payload["compacted_at"] = now
        old_payload["updated_at"] = now
        self._atomic_write_json(session.session_path, old_payload)

        session_records = list(conversation.get("sessions") or [])
        updated_records: list[dict[str, Any]] = []
        found_old = False
        for record in session_records:
            if isinstance(record, dict) and record.get("session_id") == session.session_id:
                updated = dict(record)
                updated["status"] = "compacted"
                updated["next_session_id"] = new_session_id
                updated["compacted_at"] = now
                updated_records.append(updated)
                found_old = True
            else:
                updated_records.append(record)
        if not found_old:
            updated_records.append(
                {
                    "session_id": session.session_id,
                    "status": "compacted",
                    "previous_session_id": old_payload.get("previous_session_id"),
                    "next_session_id": new_session_id,
                    "created_at": old_payload.get("created_at"),
                    "compacted_at": now,
                }
            )
        updated_records.append(
            {
                "session_id": new_session_id,
                "status": "active",
                "previous_session_id": session.session_id,
                "next_session_id": None,
                "created_at": now,
                "compacted_at": None,
            }
        )
        conversation["active_session_id"] = new_session_id
        conversation["updated_at"] = now
        conversation["sessions"] = updated_records
        self._atomic_write_json(session.conversation_path, conversation)
        return new_session

    def _session_ref(self, conversation_id: str, session_id: str) -> SessionRef:
        conversation_dir = self.conversations_dir / conversation_id
        session_dir = conversation_dir / "sessions" / session_id
        return SessionRef(
            conversation_id=conversation_id,
            session_id=session_id,
            conversation_dir=conversation_dir,
            conversation_path=conversation_dir / "conversation.json",
            session_dir=session_dir,
            session_path=session_dir / "session.json",
            memory_dir=session_dir / "memory",
            runs_dir=session_dir / "runs",
        )

    def _load_conversation(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("conversation.json root must be an object")
        conversation_id = payload.get("conversation_id")
        active_session_id = payload.get("active_session_id")
        if not isinstance(conversation_id, str) or not isinstance(active_session_id, str):
            raise ValueError("conversation.json must include conversation_id and active_session_id")
        return payload

    def _load_session(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("session.json root must be an object")
        if not isinstance(payload.get("conversation_id"), str) or not isinstance(payload.get("session_id"), str):
            raise ValueError("session.json must include conversation_id and session_id")
        return payload

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp_path, path)


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


def _new_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-{uuid.uuid4().hex[:8]}"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _session_payload(
    *,
    conversation_id: str,
    session_id: str,
    status: str,
    previous_session_id: str | None,
    next_session_id: str | None,
    compacted_at: str | None,
    messages: list[dict[str, Any]],
    created_at: str,
    updated_at: str,
) -> dict[str, Any]:
    return {
        "conversation_id": conversation_id,
        "session_id": session_id,
        "status": status,
        "previous_session_id": previous_session_id,
        "next_session_id": next_session_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "compacted_at": compacted_at,
        "messages": messages,
    }


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
