from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from coding_agent.context.context import Context
from coding_agent.session import SessionRuntime


@dataclass
class AgentTurnContext:
    run_id: str = ""
    session_id: str = ""
    turn_index: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    context_entity: Context | None = None
    session_runtime: SessionRuntime | None = None
    llm_payload: dict[str, Any] = field(default_factory=dict)
    llm_response: dict[str, Any] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    state: dict[str, Any] = field(default_factory=dict)


class AgentLifecycleHook:
    def on_turn_start(self, ctx: AgentTurnContext) -> None:
        pass

    def pre_llm(self, ctx: AgentTurnContext) -> None:
        pass

    def after_llm(self, ctx: AgentTurnContext) -> None:
        pass

    def pre_tool(self, ctx: AgentTurnContext, tool_name: str, args: dict[str, Any]) -> None:
        pass

    def after_tool(self, ctx: AgentTurnContext, tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        pass

    def on_turn_end(self, ctx: AgentTurnContext) -> None:
        pass


class AgentLifecycleBus:
    def __init__(self, hooks: list[AgentLifecycleHook] | None = None) -> None:
        self.hooks = list(hooks or [])

    def on_turn_start(self, ctx: AgentTurnContext) -> None:
        for hook in self.hooks:
            hook.on_turn_start(ctx)

    def pre_llm(self, ctx: AgentTurnContext) -> None:
        for hook in self.hooks:
            hook.pre_llm(ctx)

    def after_llm(self, ctx: AgentTurnContext) -> None:
        for hook in self.hooks:
            hook.after_llm(ctx)

    def pre_tool(self, ctx: AgentTurnContext, tool_name: str, args: dict[str, Any]) -> None:
        for hook in self.hooks:
            hook.pre_tool(ctx, tool_name, args)

    def after_tool(self, ctx: AgentTurnContext, tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        for hook in self.hooks:
            hook.after_tool(ctx, tool_name, args, result)

    def on_turn_end(self, ctx: AgentTurnContext) -> None:
        for hook in self.hooks:
            hook.on_turn_end(ctx)
