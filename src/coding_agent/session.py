from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class SessionStore:
    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root)
        self.sessions_dir = self.project_root / ".coding-agent" / "sessions"

    def save(self, records: list[dict[str, Any]]) -> Path:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        path = self.sessions_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def load(path: Path | str) -> list[dict[str, Any]]:
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("Session JSON root must be a list")
        return records

    def latest(self) -> Path | None:
        if not self.sessions_dir.exists():
            return None
        sessions = sorted(self.sessions_dir.glob("*.json"))
        if not sessions:
            return None
        return sessions[-1]

    def load_latest(self) -> list[dict[str, Any]] | None:
        path = self.latest()
        if path is None:
            return None
        return self.load(path)
