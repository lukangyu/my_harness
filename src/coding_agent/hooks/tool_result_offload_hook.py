from __future__ import annotations

import json
from typing import Any

from coding_agent.memory.store import MemoryStore
from coding_agent.orchestrator.lifecycle import AgentLifecycleHook, AgentTurnContext
from coding_agent.telemetry.logger import TelemetryLogger


class ToolResultOffloadHook(AgentLifecycleHook):
    def __init__(
        self,
        memory_store: MemoryStore,
        *,
        telemetry: TelemetryLogger | None = None,
        max_inline_chars: int = 4000,
    ) -> None:
        self.memory_store = memory_store
        self.telemetry = telemetry
        self.max_inline_chars = max_inline_chars

    def after_tool(
        self,
        ctx: AgentTurnContext,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        content = json.dumps(result, ensure_ascii=False)
        offload = self.memory_store.offload_tool_result(
            tool=tool_name,
            content=content,
            max_inline_chars=self.max_inline_chars,
        )
        if not offload.get("offloaded"):
            return None
        self._event(
            "tool.result.offload",
            f"工具 {tool_name} 的长输出已转存到文件",
            {"tool": tool_name, "path": offload.get("path"), "original_chars": offload.get("original_chars")},
        )
        return {
            "ok": result.get("ok"),
            "offloaded": True,
            "tool": tool_name,
            "preview": offload["content"],
            "path": offload["path"],
            "original_chars": offload["original_chars"],
            "instruction": f"Full tool output is archived at {offload['path']}. Use read_file if exact output is needed.",
        }

    def _event(self, event: str, message_zh: str, metadata: dict[str, Any]) -> None:
        if self.telemetry is None:
            return
        self.telemetry.event(
            event,
            message_zh,
            function="ToolResultOffloadHook.after_tool",
            phase="tool",
            metadata=metadata,
        )
