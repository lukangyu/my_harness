from __future__ import annotations

from coding_agent.context.compressor import ContextCompressor
from coding_agent.context.context import CompactClient
from coding_agent.memory.store import MemoryStore
from coding_agent.orchestrator.lifecycle import AgentLifecycleHook, AgentTurnContext


class ContextCompactionHook(AgentLifecycleHook):
    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        compact_client: CompactClient,
    ) -> None:
        self.memory_store = memory_store
        self.compact_client = compact_client
        self.compressor = ContextCompressor(
            memory_store=memory_store,
            compact_client=compact_client,
        )

    def pre_llm(self, ctx: AgentTurnContext) -> None:
        context = ctx.context_entity
        if context is None:
            return
        if self.compressor.should_compress(context):
            result = self.compressor.compress(context)
            if result.compacted and ctx.session_runtime is not None and result.summary_message is not None:
                next_session = ctx.session_runtime.store.compact_session(
                    ctx.session_runtime.current,
                    seed_messages=result.remaining_messages or [],
                    summary=result.summary,
                )
                ctx.session_runtime.current = next_session
