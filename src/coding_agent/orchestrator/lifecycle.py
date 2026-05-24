from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from coding_agent.context.context import Context
from coding_agent.session import SessionRuntime


class ToolVeto(Exception):
    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(str(result.get("error") or result.get("code") or "tool vetoed"))
        self.result = result


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


LifecyclePhase = Literal["on_turn_start", "pre_llm", "after_llm", "pre_tool", "after_tool", "on_turn_end"]


@dataclass(frozen=True)
class HookRegistration:
    phase: LifecyclePhase
    hook: "AgentLifecycleHook"
    order: int = 1000
    sequence: int = 0


class AgentLifecycleHook:
    def on_turn_start(self, ctx: AgentTurnContext) -> None:
        pass

    def pre_llm(self, ctx: AgentTurnContext) -> None:
        pass

    def after_llm(self, ctx: AgentTurnContext) -> None:
        pass

    def pre_tool(self, ctx: AgentTurnContext, tool_name: str, args: dict[str, Any]) -> None:
        pass

    def after_tool(
        self,
        ctx: AgentTurnContext,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        return None

    def on_turn_end(self, ctx: AgentTurnContext) -> None:
        pass


class AgentLifecycleRegistry:
    def __init__(self) -> None:
        self._registrations: dict[LifecyclePhase, list[HookRegistration]] = {
            "on_turn_start": [],
            "pre_llm": [],
            "after_llm": [],
            "pre_tool": [],
            "after_tool": [],
            "on_turn_end": [],
        }
        self._sequence = 0

    def add(self, phase: LifecyclePhase, hook: AgentLifecycleHook, *, order: int = 1000) -> "AgentLifecycleRegistry":
        self._registrations[phase].append(
            HookRegistration(phase=phase, hook=hook, order=order, sequence=self._sequence)
        )
        self._sequence += 1
        return self

    def hooks_for(self, phase: LifecyclePhase) -> list[AgentLifecycleHook]:
        return [
            registration.hook
            for registration in sorted(
                self._registrations[phase],
                key=lambda registration: (registration.order, registration.sequence),
            )
        ]

    def registrations(self) -> list[HookRegistration]:
        registrations: list[HookRegistration] = []
        for phase_registrations in self._registrations.values():
            registrations.extend(phase_registrations)
        return sorted(registrations, key=lambda registration: registration.sequence)


class AgentLifecycleBus:
    def __init__(self, registry: AgentLifecycleRegistry | None = None) -> None:
        self.registry = registry or AgentLifecycleRegistry()

    def on_turn_start(self, ctx: AgentTurnContext) -> None:
        for hook in self.registry.hooks_for("on_turn_start"):
            hook.on_turn_start(ctx)

    def pre_llm(self, ctx: AgentTurnContext) -> None:
        for hook in self.registry.hooks_for("pre_llm"):
            hook.pre_llm(ctx)

    def after_llm(self, ctx: AgentTurnContext) -> None:
        for hook in self.registry.hooks_for("after_llm"):
            hook.after_llm(ctx)

    def pre_tool(self, ctx: AgentTurnContext, tool_name: str, args: dict[str, Any]) -> None:
        for hook in self.registry.hooks_for("pre_tool"):
            hook.pre_tool(ctx, tool_name, args)

    def after_tool(
        self,
        ctx: AgentTurnContext,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        current = result
        for hook in self.registry.hooks_for("after_tool"):
            next_result = hook.after_tool(ctx, tool_name, args, current)
            if next_result is not None:
                current = next_result
        return current

    def on_turn_end(self, ctx: AgentTurnContext) -> None:
        for hook in self.registry.hooks_for("on_turn_end"):
            hook.on_turn_end(ctx)
