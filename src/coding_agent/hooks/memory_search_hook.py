from __future__ import annotations

from typing import Any

from coding_agent.context.virtual_tools import AUTO_MEMORY_SEARCH, synthetic_tool_pair
from coding_agent.memory.store import MemoryStore
from coding_agent.orchestrator.lifecycle import AgentLifecycleHook, AgentTurnContext
from coding_agent.telemetry.logger import TelemetryLogger


class MemorySearchHook(AgentLifecycleHook):
    def __init__(
        self,
        memory_store: MemoryStore,
        *,
        max_results: int = 3,
        telemetry: TelemetryLogger | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.max_results = max_results
        self.telemetry = telemetry

    def pre_llm(self, ctx: AgentTurnContext) -> None:
        context = ctx.context_entity
        if context is None:
            return
        query = _query_from_task(context.task)
        if not query:
            return
        results = self.memory_store.search_long_term_memories(query, max_results=self.max_results)
        if not results:
            return
        payload = {
            "kind": "memory_search_results",
            "schema_version": 1,
            "query": query,
            "results": results,
        }
        ctx.messages.extend(
            synthetic_tool_pair(
                name=AUTO_MEMORY_SEARCH,
                call_id=f"call_auto_memory_search_{ctx.turn_index or 0}",
                arguments={"query": query},
                tag="memory_search_results",
                payload=payload,
            )
        )
        if self.telemetry is not None:
            self.telemetry.event(
                "memory.auto_search",
                "自动检索长期记忆并注入上下文",
                function="MemorySearchHook.pre_llm",
                phase="memory",
                metadata={
                    "query": query,
                    "result_count": len(results),
                    "sources": [item.get("source") for item in results if isinstance(item, dict)],
                },
            )


def _query_from_task(task: Any) -> str:
    if not isinstance(task, str):
        return ""
    return " ".join(task.strip().split())[:100]
