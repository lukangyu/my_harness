from __future__ import annotations

from typing import Any

from coding_agent.memory.store import MemoryStore
from coding_agent.orchestrator.lifecycle import AgentLifecycleHook, AgentTurnContext


class MemoryProjectionHook(AgentLifecycleHook):
    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store

    def after_tool(
        self,
        ctx: AgentTurnContext,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        self.memory_store.record_tool_result(
            tool=tool_name,
            arguments=args,
            result=result,
        )
