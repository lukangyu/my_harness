from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class DialogArchiveStore:
    def __init__(self, project_root: Path, dialog_dir: Path) -> None:
        self.project_root = project_root
        self.dialog_dir = dialog_dir

    def archive_messages(self, messages: list[dict[str, Any]]) -> Path | None:
        if not messages:
            return None
        self.dialog_dir.mkdir(parents=True, exist_ok=True)
        path = self.dialog_dir / f"{time.strftime('%Y-%m-%d', time.localtime())}-{uuid.uuid4().hex[:8]}.jsonl"
        with path.open("w", encoding="utf-8") as file:
            for message in messages:
                file.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
        return path
