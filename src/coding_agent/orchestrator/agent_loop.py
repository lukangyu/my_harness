from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from coding_agent.context.assembler import ContextAssembler
from coding_agent.context.context import UsageStats, WorkspaceContextOptions
from coding_agent.context.prompt_builder import PromptBuilder, create_default_prompt_builder
from coding_agent.execution.executor import ToolExecutor
from coding_agent.execution.tools import ToolRegistry
from coding_agent.memory.store import MemoryStore
from coding_agent.orchestrator.lifecycle import AgentLifecycleBus, AgentLifecycleHook, AgentTurnContext
from coding_agent.runtime_events import RuntimeEvent
from coding_agent.session import SessionRuntime
from coding_agent.telemetry.logger import TelemetryLogger


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        ...


@dataclass
class AgentResult:
    final_answer: str
    messages: list[dict[str, Any]]
    conversation_messages: list[dict[str, Any]]
    reached_max_steps: bool = False
    usage: UsageStats | None = None
    attempts: int = 0
    tool_steps: int = 0
    last_tool: str | None = None
    stop_reason: str = "final_answer"


class AgentLoop:
    def __init__(
        self,
        client: ChatClient,
        tools: ToolRegistry,
        max_steps: int,
        cwd: Path,
        context_options: WorkspaceContextOptions | None = None,
        recent_message_tokens: int = 12000,
        on_tool_call: Callable[[dict[str, Any]], None] | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        on_runtime_event: Callable[[RuntimeEvent], None] | None = None,
        telemetry: TelemetryLogger | None = None,
        memory_store: MemoryStore | None = None,
        lifecycle_hooks: list[AgentLifecycleHook] | None = None,
        prompt_builder: PromptBuilder | None = None,
        context_assembler: ContextAssembler | None = None,
        session_runtime: SessionRuntime | None = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.max_steps = max_steps
        self.on_tool_call = on_tool_call
        self.on_progress = on_progress
        self.on_runtime_event = on_runtime_event
        self.telemetry = telemetry
        self.memory_store = memory_store
        self.lifecycle_hooks = list(lifecycle_hooks or [])
        self.lifecycle = AgentLifecycleBus(self.lifecycle_hooks)
        self.cwd = cwd
        self.context_options = context_options or WorkspaceContextOptions()
        self.recent_message_tokens = recent_message_tokens
        self.prompt_builder = prompt_builder or create_default_prompt_builder()
        self.context_assembler = context_assembler or ContextAssembler(
            cwd=cwd,
            options=self.context_options,
            memory_store=memory_store,
        )
        self.session_runtime = session_runtime
        self.tool_executor = ToolExecutor(
            tools,
            lifecycle_hooks=self.lifecycle_hooks,
            on_tool_call=on_tool_call,
            on_runtime_event=on_runtime_event,
            telemetry=telemetry,
            memory_store=memory_store,
        )

    def run(
        self,
        task: str,
        prior_messages: list[dict[str, Any]] | None = None,
        mode: str = "run",
    ) -> AgentResult:
        self._event("agent.run.start", "AgentLoop 开始执行任务", "run", {"mode": mode, "task_length": len(task)})
        tool_schemas = self.tools.schemas()
        with self._span("构建上下文", "context", {"tool_count": len(tool_schemas)}):
            context = self.context_assembler.build(
                task=task,
                prior_messages=list(prior_messages or []),
                tool_schemas=tool_schemas,
                mode=mode,
            )
        self._event(
            "context.built",
            "上下文构建完成",
            "context",
            {
                "full_context_key": context.full_context_key(),
                "recent_messages": len(context.history_frames()),
                "workspace_fingerprint": context.workspace_snapshot.fingerprint(),
            },
        )
        self._runtime_event(
            RuntimeEvent(
                type="context.built",
                message="上下文已组装",
                metadata={
                    "tool_count": len(tool_schemas),
                    "recent_messages": len(context.history_frames()),
                    "memory_anchor": any(frame.kind == "memory" for frame in context.context_frames()),
                    "handoff_memo": any(frame.kind == "handoff" for frame in context.context_frames()),
                },
            )
        )
        conversation_messages = context.raw_messages()
        conversation_messages.append({"role": "user", "content": context.task})
        current_turn_tail: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []
        usage: UsageStats | None = None
        attempts = 0
        tool_steps = 0
        last_tool: str | None = None

        for step in range(1, self.max_steps + 1):
            turn_ctx = AgentTurnContext(
                turn_index=step,
                messages=[],
                tool_schemas=tool_schemas,
                context_entity=context,
                session_runtime=self.session_runtime,
            )
            self.lifecycle.on_turn_start(turn_ctx)
            self.lifecycle.pre_llm(turn_ctx)
            with self._span("渲染 prompt messages", "prompt"):
                messages = self.prompt_builder.build_final_messages(context)
            extra_messages = deepcopy(turn_ctx.messages)
            current_task_message = _pop_current_task_message(messages)
            if current_task_message is not None:
                messages.extend(extra_messages)
                messages.append(current_task_message)
            else:
                messages.extend(extra_messages)
            messages.extend(deepcopy(current_turn_tail))
            turn_ctx.messages = messages
            self._event(
                "agent.step.start",
                f"开始第 {step} 轮 AgentLoop",
                "agent_step",
                {"step": step, "message_count": len(messages)},
            )
            with self._span("调用大模型", "llm", {"step": step, "message_count": len(messages)}):
                response = self.client.chat(messages, tool_schemas)
            turn_ctx.llm_response = response
            self.lifecycle.after_llm(turn_ctx)
            attempts += 1
            response_usage = response.get("usage")
            if response_usage is not None:
                usage = response_usage
            self._progress({"type": "model_attempt", "attempts": attempts, "step": step})
            assistant_message = response["message"]
            messages.append(assistant_message)
            current_turn_tail.append(deepcopy(assistant_message))
            conversation_messages.append(deepcopy(assistant_message))

            tool_calls = assistant_message.get("tool_calls") or []
            self._event(
                "assistant.message",
                "收到 assistant 消息",
                "llm",
                {
                    "step": step,
                    "content_length": len(assistant_message.get("content") or ""),
                    "reasoning_length": len(assistant_message.get("reasoning_content") or ""),
                    "tool_call_count": len(tool_calls),
                },
            )
            if not tool_calls:
                self._event("agent.run.end", "AgentLoop 获得最终答案并结束", "run", {"step": step})
                return AgentResult(
                    final_answer=assistant_message.get("content") or "",
                    messages=messages,
                    conversation_messages=conversation_messages,
                    usage=usage,
                    attempts=attempts,
                    tool_steps=tool_steps,
                    last_tool=last_tool,
                    stop_reason="final_answer",
                )

            for tool_call in tool_calls:
                with self._span("生成 tool message", "tool", {"step": step}):
                    tool_message = self.tool_executor.execute(tool_call, turn_ctx)
                tool_steps += 1
                last_tool = tool_message.get("name")
                self._progress(
                    {
                        "type": "tool_step",
                        "tool_steps": tool_steps,
                        "tool": last_tool,
                        "step": step,
                    }
                )
                messages.append(tool_message)
                current_turn_tail.append(deepcopy(tool_message))
                conversation_messages.append(deepcopy(tool_message))
            self.lifecycle.on_turn_end(turn_ctx)

        self._event("agent.max_steps", "AgentLoop 达到最大步数后停止", "run", {"max_steps": self.max_steps})
        return AgentResult(
            final_answer="Stopped after reaching max steps.",
            messages=messages,
            conversation_messages=conversation_messages,
            reached_max_steps=True,
            usage=usage,
            attempts=attempts,
            tool_steps=tool_steps,
            last_tool=last_tool,
            stop_reason="max_steps",
        )

    def _event(self, event: str, message_zh: str, phase: str, metadata: dict[str, Any] | None = None) -> None:
        if self.telemetry is None:
            return
        self.telemetry.event(
            event,
            message_zh,
            function="AgentLoop.run",
            phase=phase,
            metadata=metadata,
        )

    def _progress(self, event: dict[str, Any]) -> None:
        if self.on_progress is not None:
            self.on_progress(event)

    def _runtime_event(self, event: RuntimeEvent) -> None:
        if self.on_runtime_event is not None:
            self.on_runtime_event(event)

    def _span(self, name: str, phase: str, metadata: dict[str, Any] | None = None) -> Any:
        if self.telemetry is None:
            return _NullSpan()
        return self.telemetry.span(name, function="AgentLoop.run", phase=phase, metadata=metadata)


class _NullSpan:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False


def _pop_current_task_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not messages:
        return None
    last_message = messages[-1]
    content = last_message.get("content")
    if last_message.get("role") != "user" or not isinstance(content, str):
        return None
    if "<current_task" in content:
        return messages.pop()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and payload.get("kind") == "current_task":
        return messages.pop()
    return None
