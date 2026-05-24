from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_SCRATCHPAD: dict[str, Any] = {
    "project_goal": "",
    "user_preferences": [],
    "confirmed_decisions": [],
    "modified_files": [],
    "read_files": [],
    "known_issues": [],
    "active_todos": [],
    "last_verified_commands": [],
}


class ScratchpadStore:
    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.path = memory_dir / "scratchpad.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return deepcopy(DEFAULT_SCRATCHPAD)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return deepcopy(DEFAULT_SCRATCHPAD)
        if not isinstance(data, dict):
            return deepcopy(DEFAULT_SCRATCHPAD)
        scratchpad = deepcopy(DEFAULT_SCRATCHPAD)
        scratchpad.update(data)
        return scratchpad

    def save(self, scratchpad: dict[str, Any]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(normalize_scratchpad(scratchpad), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def normalize_scratchpad(scratchpad: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(DEFAULT_SCRATCHPAD)
    normalized.update(scratchpad)
    for key in (
        "user_preferences",
        "confirmed_decisions",
        "modified_files",
        "read_files",
        "known_issues",
        "active_todos",
        "last_verified_commands",
    ):
        if not isinstance(normalized.get(key), list):
            normalized[key] = []
    if not isinstance(normalized.get("project_goal"), str):
        normalized["project_goal"] = ""
    return normalized
