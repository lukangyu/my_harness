from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ToolIndexStore:
    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.path = memory_dir / "tool_index.jsonl"

    def append(self, record: dict[str, Any]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        enriched = {"timestamp": _now(), **record}
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(enriched, ensure_ascii=False, default=str) + "\n")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
