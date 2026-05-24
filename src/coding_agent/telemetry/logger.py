from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import json
import os
import time
import uuid


IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".coding-agent",
}


class TelemetryLogger:
    def __init__(
        self,
        logs_dir: Path | str,
        workspace_root: Path | str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.workspace_root = Path(workspace_root).resolve() if workspace_root is not None else None
        self.run_id = run_id or uuid.uuid4().hex

    def event(
        self,
        event: str,
        message_zh: str,
        *,
        function: str,
        phase: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._append_jsonl(
            "events.jsonl",
            {
                "kind": "event",
                "timestamp": _now_ms(),
                "run_id": self.run_id,
                "event": event,
                "message_zh": message_zh,
                "function": function,
                "phase": phase,
                "metadata": metadata or {},
            },
        )

    @contextmanager
    def span(
        self,
        name: str,
        *,
        function: str,
        phase: str,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        span_id = uuid.uuid4().hex
        started = time.perf_counter()
        ok = False
        error: str | None = None
        try:
            yield span_id
            ok = True
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)
            record: dict[str, Any] = {
                "kind": "span",
                "timestamp": _now_ms(),
                "run_id": self.run_id,
                "span_id": span_id,
                "name": name,
                "function": function,
                "phase": phase,
                "duration_ms": duration_ms,
                "ok": ok,
                "metadata": metadata or {},
            }
            if error:
                record["error"] = error
            self._append_jsonl("events.jsonl", record)

    def workspace_snapshot(
        self,
        *,
        message_zh: str,
        function: str,
        phase: str,
        root: Path | str | None = None,
        max_entries: int = 200,
    ) -> None:
        workspace_root = Path(root).resolve() if root is not None else self.workspace_root
        if workspace_root is None:
            return
        snapshot = build_workspace_snapshot(workspace_root, max_entries=max_entries)
        self.event(
            "workspace.snapshot",
            message_zh,
            function=function,
            phase=phase,
            metadata=snapshot,
        )

    def _append_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        path = self.logs_dir / filename
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def build_workspace_snapshot(root: Path, *, max_entries: int = 200) -> dict[str, Any]:
    files: list[str] = []
    directories: list[str] = []
    total_files = 0
    total_directories = 0
    truncated = False

    if not root.exists():
        return {
            "root": str(root),
            "exists": False,
            "files": [],
            "directories": [],
            "total_files": 0,
            "total_directories": 0,
            "truncated": False,
        }

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        relative_dir = current.relative_to(root)
        if _is_ignored(relative_dir):
            dirnames[:] = []
            continue
        dirnames[:] = sorted(name for name in dirnames if not _is_ignored(relative_dir / name))
        if relative_dir.parts:
            total_directories += 1
            if len(directories) + len(files) < max_entries:
                directories.append(relative_dir.as_posix())
            else:
                truncated = True
        for filename in sorted(filenames):
            relative_file = (current / filename).relative_to(root)
            if _is_ignored(relative_file):
                continue
            total_files += 1
            if len(directories) + len(files) < max_entries:
                files.append(relative_file.as_posix())
            else:
                truncated = True

    return {
        "root": str(root),
        "exists": True,
        "files": files,
        "directories": directories,
        "total_files": total_files,
        "total_directories": total_directories,
        "truncated": truncated,
    }


def _is_ignored(relative: Path) -> bool:
    return any(part in IGNORED_DIRS or part.startswith(".pytest") for part in relative.parts)


def _now_ms() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + f".{int(time.time_ns() / 1_000_000) % 1000:03d}"
