from __future__ import annotations

import json
import os
import subprocess
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

from coding_agent.checkpoint.hashing import file_state
from coding_agent.execution.sandbox import WorkspaceSandbox


class CheckpointConflict(RuntimeError):
    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(str(result.get("error") or result.get("code") or "checkpoint conflict"))
        self.result = result


class CheckpointStore:
    def __init__(self, *, conversation_dir: Path, workspace_root: Path, sandbox: WorkspaceSandbox) -> None:
        self.conversation_dir = Path(conversation_dir)
        self.workspace_root = Path(workspace_root).resolve()
        self.sandbox = sandbox
        self.checkpoint_path = self.conversation_dir / "checkpoints" / "checkpoint.json"

    def load(self) -> dict[str, Any]:
        if not self.checkpoint_path.exists():
            return _empty_payload()
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("checkpoint.json root must be an object")
        payload.setdefault("version", 1)
        payload.setdefault("workspace", {})
        payload.setdefault("files", {})
        return payload

    def save_atomic(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _now()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.checkpoint_path.with_name(f"{self.checkpoint_path.name}.tmp")
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp_path, self.checkpoint_path)

    def refresh_workspace(self, *, run_id: str = "", session_id: str = "") -> None:
        payload = self.load()
        workspace = self.current_workspace_state()
        workspace["last_seen_run_id"] = run_id
        workspace["last_seen_session_id"] = session_id
        workspace["updated_at"] = _now()
        payload["workspace"] = workspace
        self.save_atomic(payload)

    def record_file(self, path: str, *, source: str, run_id: str = "", session_id: str = "") -> None:
        state = file_state(path, self.sandbox)
        record = {
            **state.to_dict(),
            "source": source,
            "last_seen_run_id": run_id,
            "last_seen_session_id": session_id,
            "updated_at": _now(),
        }
        payload = self.load()
        payload.setdefault("files", {})[state.path] = record
        self.save_atomic(payload)

    def record_deleted(self, path: str, *, source: str, run_id: str = "", session_id: str = "") -> None:
        state = file_state(path, self.sandbox)
        record = {
            "path": state.path,
            "exists": False,
            "content_hash": None,
            "size": None,
            "mtime_ns": None,
            "source": source,
            "last_seen_run_id": run_id,
            "last_seen_session_id": session_id,
            "updated_at": _now(),
        }
        payload = self.load()
        payload.setdefault("files", {})[state.path] = record
        self.save_atomic(payload)

    def verify_workspace_not_drifted(self) -> None:
        saved = self.load().get("workspace") or {}
        if not saved:
            return
        current = self.current_workspace_state()
        for key in ("repo_root", "branch", "head_commit"):
            saved_value = saved.get(key)
            current_value = current.get(key)
            if saved_value is not None and current_value is not None and saved_value != current_value:
                raise CheckpointConflict(
                    {
                        "ok": False,
                        "code": "workspace_drift_detected",
                        "field": key,
                        "expected": saved_value,
                        "actual": current_value,
                        "error": f"Workspace {key} changed after it was last checkpointed.",
                        "instruction": "Stop this write, inspect the current workspace state, and re-read affected files before modifying them.",
                    }
                )

    def verify_file_not_drifted(self, path: str) -> None:
        state = file_state(path, self.sandbox)
        record = self.load().get("files", {}).get(state.path)
        if record is None:
            if state.exists:
                raise CheckpointConflict(
                    {
                        "ok": False,
                        "code": "file_not_seen",
                        "path": state.path,
                        "error": "Existing file has not been read in this conversation checkpoint.",
                        "instruction": "Read this file with read_file before modifying it.",
                    }
                )
            return
        if bool(record.get("exists")) != state.exists or record.get("content_hash") != state.content_hash:
            raise CheckpointConflict(
                {
                    "ok": False,
                    "code": "file_drift_detected",
                    "path": state.path,
                    "error": "File changed after it was last read.",
                    "instruction": "Re-read this file with read_file before modifying it again.",
                }
            )

    def verify_file_absent_for_add(self, path: str) -> None:
        state = file_state(path, self.sandbox)
        if state.exists:
            raise CheckpointConflict(
                {
                    "ok": False,
                    "code": "file_already_exists",
                    "path": state.path,
                    "error": "Patch attempted to add a file that already exists.",
                    "instruction": "Read the existing file and generate an update patch instead of Add File.",
                }
            )

    def current_workspace_state(self) -> dict[str, Any]:
        git = _git_identity(self.workspace_root)
        fingerprint_input = json.dumps(git, sort_keys=True, ensure_ascii=False)
        return {
            "repo_root": str(self.workspace_root),
            **git,
            "fingerprint": sha256(fingerprint_input.encode("utf-8")).hexdigest(),
        }


def _git_identity(root: Path) -> dict[str, Any]:
    repo_root = _git(root, "rev-parse", "--show-toplevel")
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    head_commit = _git(root, "rev-parse", "HEAD")
    status = _git(root, "status", "--short")
    return {
        "git_repo_root": repo_root,
        "branch": branch,
        "head_commit": head_commit,
        "git_status": status,
    }


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _empty_payload() -> dict[str, Any]:
    now = _now()
    return {"version": 1, "created_at": now, "updated_at": now, "workspace": {}, "files": {}}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
