from __future__ import annotations

import json
from typing import Any, Callable

from coding_agent.memory.store import MemoryStore
from coding_agent.orchestrator.lifecycle import AgentLifecycleBus, AgentLifecycleHook, AgentTurnContext
from coding_agent.runtime_events import RuntimeEvent
from coding_agent.telemetry.logger import TelemetryLogger
from coding_agent.execution.tools import ToolRegistry


class ToolExecutor:
    def __init__(
        self,
        tools: ToolRegistry,
        *,
        lifecycle_hooks: list[AgentLifecycleHook] | None = None,
        on_tool_call: Callable[[dict[str, Any]], None] | None = None,
        on_runtime_event: Callable[[RuntimeEvent], None] | None = None,
        telemetry: TelemetryLogger | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.tools = tools
        self.lifecycle = AgentLifecycleBus(lifecycle_hooks)
        self.on_tool_call = on_tool_call
        self.on_runtime_event = on_runtime_event
        self.telemetry = telemetry
        self.memory_store = memory_store

    def execute(self, tool_call: dict[str, Any], ctx: AgentTurnContext) -> dict[str, Any]:
        if self.on_tool_call is not None:
            self.on_tool_call(tool_call)
        function = tool_call.get("function", {})
        name = function.get("name", "")
        raw_arguments = function.get("arguments") or "{}"
        self._event(
            "agent.tool_message.start",
            f"开始处理模型请求的工具调用 {name}",
            "tool",
            {"tool": name, "arguments_length": len(raw_arguments)},
        )
        arguments: dict[str, Any] = {}
        try:
            parsed_arguments = json.loads(raw_arguments)
            if not isinstance(parsed_arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            arguments = parsed_arguments
            self.lifecycle.pre_tool(ctx, name, arguments)
            result = self.tools.call(name, arguments)
        except (json.JSONDecodeError, ValueError) as exc:
            result = {"ok": False, "error": f"Invalid JSON arguments: {exc}"}
        self.lifecycle.after_tool(ctx, name, arguments, result)
        self._runtime_event(
            RuntimeEvent(
                type="tool.result",
                message=f"工具 {name} 执行完成",
                metadata=tool_result_metadata(name, result),
            )
        )
        tool_content = json.dumps(result, ensure_ascii=False)
        if self.memory_store is not None:
            offload = self.memory_store.offload_tool_result(tool=name, content=tool_content)
            tool_content = offload["content"]
            if offload.get("offloaded"):
                self._event(
                    "tool.result.offload",
                    f"工具 {name} 的长输出已转存到文件",
                    "tool",
                    {"tool": name, "path": offload.get("path"), "original_chars": offload.get("original_chars")},
                )
        self._event(
            "agent.tool_message.end",
            f"工具调用 {name} 的 tool message 已生成",
            "tool",
            {"tool": name, "ok": result.get("ok"), "content_length": len(tool_content)},
        )
        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "name": name,
            "content": tool_content,
        }

    def _runtime_event(self, event: RuntimeEvent) -> None:
        if self.on_runtime_event is not None:
            self.on_runtime_event(event)

    def _event(self, event: str, message_zh: str, phase: str, metadata: dict[str, Any] | None = None) -> None:
        if self.telemetry is None:
            return
        self.telemetry.event(
            event,
            message_zh,
            function="ToolExecutor.execute",
            phase=phase,
            metadata=metadata,
        )


def tool_result_metadata(tool: str, result: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "tool": tool,
        "ok": result.get("ok"),
    }
    if result.get("ok") is False:
        metadata["error"] = result.get("error") or result.get("stderr") or "工具执行失败"
    for key in ("path", "exit_code", "timed_out"):
        if key in result:
            metadata[key] = result[key]
    if "changed_files" in result:
        changed_files = result.get("changed_files") or []
        metadata["changed_files_count"] = len(changed_files) if isinstance(changed_files, list) else 0
    if "matches" in result:
        matches = result.get("matches") or []
        metadata["matches_count"] = len(matches) if isinstance(matches, list) else 0
    if "files" in result:
        files = result.get("files") or []
        metadata["files_count"] = len(files) if isinstance(files, list) else 0
    return metadata
