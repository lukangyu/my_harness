from __future__ import annotations

from pathlib import Path


class HandoffStore:
    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.path = memory_dir / "handoff.md"

    def read(self) -> str:
        if not self.path.exists():
            return ""
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def write(self, content: str) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(content.strip() + "\n", encoding="utf-8")
