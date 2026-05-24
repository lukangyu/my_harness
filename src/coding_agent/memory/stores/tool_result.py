from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any


class ToolResultStore:
    def __init__(self, project_root: Path, tool_result_dir: Path) -> None:
        self.project_root = project_root
        self.tool_result_dir = tool_result_dir

    def offload(
        self,
        *,
        tool: str,
        content: str,
        max_inline_chars: int = 4000,
    ) -> dict[str, Any]:
        if len(content) <= max_inline_chars:
            return {"content": content, "offloaded": False}
        self.tool_result_dir.mkdir(parents=True, exist_ok=True)
        path = self.tool_result_dir / f"{time.strftime('%Y%m%d-%H%M%S', time.localtime())}-{uuid.uuid4().hex[:8]}.txt"
        path.write_text(content, encoding="utf-8")
        relative = path.relative_to(self.project_root).as_posix()
        snippet = content[:max_inline_chars]
        return {
            "content": (
                f"{snippet}\n"
                f"... [完整 tool 输出已转存到 {relative}，当前上下文只保留前 {max_inline_chars} 字符]"
            ),
            "offloaded": True,
            "path": relative,
            "original_chars": len(content),
            "tool": tool,
        }
