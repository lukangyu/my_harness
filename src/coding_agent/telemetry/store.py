from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import time
import uuid


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunArtifact:
    run_id: str
    run_dir: Path
    agent_context_dir: Path
    audit_dir: Path
    task_state_path: Path
    events_path: Path
    report_path: Path
    debug_dir: Path
    dialog_dir: Path
    tool_result_dir: Path


@dataclass
class TaskState:
    run_id: str
    task: str
    mode: str
    workspace_root: str
    status: str = "running"
    attempts: int = 0
    tool_steps: int = 0
    last_tool: str | None = None
    stop_reason: str | None = None
    final_answer: str = ""
    error: str | None = None
    schema_version: int = SCHEMA_VERSION
    updated_at: str = field(default_factory=lambda: _now())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task": self.task,
            "mode": self.mode,
            "workspace_root": self.workspace_root,
            "status": self.status,
            "attempts": self.attempts,
            "tool_steps": self.tool_steps,
            "last_tool": self.last_tool,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
            "error": self.error,
            "updated_at": self.updated_at,
        }


class RunStore:
    def __init__(self, project_root: Path | str, *, runs_root: Path | str | None = None) -> None:
        self.project_root = Path(project_root)
        self.runs_dir = Path(runs_root) if runs_root is not None else self.project_root / ".coding-agent" / "runs"
        self.index_path = self.runs_dir / "index.jsonl"

    def start_run(self, *, task: str, mode: str, workspace_root: Path | str) -> tuple[RunArtifact, TaskState]:
        run_id = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-{uuid.uuid4().hex[:8]}"
        run_dir = self.runs_dir / run_id
        agent_context_dir = run_dir / "agent_context"
        audit_dir = run_dir / "audit"
        artifact = RunArtifact(
            run_id=run_id,
            run_dir=run_dir,
            agent_context_dir=agent_context_dir,
            audit_dir=audit_dir,
            task_state_path=audit_dir / "task_state.json",
            events_path=audit_dir / "events.jsonl",
            report_path=audit_dir / "report.json",
            debug_dir=audit_dir / "debug",
            dialog_dir=agent_context_dir / "dialog",
            tool_result_dir=agent_context_dir / "tool_result",
        )
        for path in (artifact.audit_dir, artifact.debug_dir, artifact.dialog_dir, artifact.tool_result_dir):
            path.mkdir(parents=True, exist_ok=True)
        state = TaskState(run_id=run_id, task=task, mode=mode, workspace_root=str(Path(workspace_root).resolve()))
        self.write_task_state(artifact, state)
        self._append_index({"run_id": run_id, "task": task, "mode": mode, "status": state.status, "started_at": state.updated_at})
        return artifact, state

    def write_task_state(self, artifact: RunArtifact, state: TaskState) -> None:
        state.updated_at = _now()
        artifact.run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(artifact.task_state_path, state.to_dict())

    def write_report(
        self,
        artifact: RunArtifact,
        state: TaskState,
        *,
        session_path: Path | str | None,
        usage: Any = None,
        files_changed: list[str] | None = None,
    ) -> None:
        report = {
            "schema_version": SCHEMA_VERSION,
            "run_id": artifact.run_id,
            "session_path": str(session_path) if session_path is not None else None,
            "status": state.status,
            "stop_reason": state.stop_reason,
            "final_answer": state.final_answer,
            "attempts": state.attempts,
            "tool_steps": state.tool_steps,
            "last_tool": state.last_tool,
            "files_changed": files_changed or [],
            "usage": _usage_to_dict(usage),
            "prompt_metadata": _prompt_metadata(usage),
            "task_state": state.to_dict(),
            "artifact_paths": {
                "audit": {
                    "task_state": _relative(artifact.task_state_path, self.project_root),
                    "events": _relative(artifact.events_path, self.project_root),
                    "debug_dir": _relative(artifact.debug_dir, self.project_root),
                },
                "agent_context": {
                    "dialog_dir": _relative(artifact.dialog_dir, self.project_root),
                    "tool_result_dir": _relative(artifact.tool_result_dir, self.project_root),
                },
            },
            "created_at": _now(),
        }
        _write_json(artifact.report_path, report)
        self._append_index({"run_id": artifact.run_id, "status": state.status, "stop_reason": state.stop_reason, "ended_at": report["created_at"]})

    def _append_index(self, record: dict[str, Any]) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"schema_version": SCHEMA_VERSION, **record}, ensure_ascii=False, default=str) + "\n")


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cached_tokens": getattr(usage, "cached_tokens", None),
        "cache_hit_ratio": getattr(usage, "cache_hit_ratio", None),
    }


def _prompt_metadata(usage: Any) -> dict[str, Any]:
    usage_dict = _usage_to_dict(usage) or {}
    return {
        "input_tokens": usage_dict.get("input_tokens"),
        "output_tokens": usage_dict.get("output_tokens"),
        "cached_tokens": usage_dict.get("cached_tokens"),
        "cache_hit_ratio": usage_dict.get("cache_hit_ratio"),
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)
