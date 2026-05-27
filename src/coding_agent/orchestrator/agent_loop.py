from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any, Callable, Protocol

from coding_agent.context.assembler import ContextAssembler
from coding_agent.context.context import UsageStats, WorkspaceContextOptions
from coding_agent.context.prompt_builder import PromptBuilder, create_default_prompt_builder
from coding_agent.execution.executor import ToolExecutor
from coding_agent.execution.tools import ToolRegistry
from coding_agent.memory.store import MemoryStore
from coding_agent.orchestrator.lifecycle import AgentLifecycleBus, AgentLifecycleRegistry, AgentTurnContext
from coding_agent.runtime_events import RuntimeEvent
from coding_agent.session import SessionRuntime, _GENERATED_PROMPT_MARKERS
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


PARALLEL_SAFE_TOOLS = {
    "list_files",
    "read_file",
    "search_text",
    "session_search",
    "run_shell",
    "start_subagent",
    "wait_subagent",
    "cancel_subagent",
}
MAX_PARALLEL_TOOLS = 4


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
        lifecycle_registry: AgentLifecycleRegistry | None = None,
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
        self.lifecycle = AgentLifecycleBus(lifecycle_registry)
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
            lifecycle=self.lifecycle,
            on_tool_call=on_tool_call,
            on_runtime_event=on_runtime_event,
            telemetry=telemetry,
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
        conversation_messages = [_sanitize_conversation_message(message) for message in context.raw_messages()]
        conversation_messages.append(_sanitize_conversation_message({"role": "user", "content": context.task}))
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
            conversation_messages.append(_sanitize_conversation_message(assistant_message))

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

            tool_messages = self._execute_tool_calls(tool_calls, turn_ctx, step)
            for tool_message in tool_messages:
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
                conversation_messages.append(_sanitize_conversation_message(tool_message))
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

    def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        turn_ctx: AgentTurnContext,
        step: int,
    ) -> list[dict[str, Any]]:
        if len(tool_calls) <= 1 or not all(_is_parallel_safe_tool(tool_call) for tool_call in tool_calls):
            return [self._execute_tool_call(tool_call, turn_ctx, step) for tool_call in tool_calls]

        max_workers = min(MAX_PARALLEL_TOOLS, len(tool_calls))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._execute_tool_call, tool_call, turn_ctx, step)
                for tool_call in tool_calls
            ]
            return [future.result() for future in futures]

    def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        turn_ctx: AgentTurnContext,
        step: int,
    ) -> dict[str, Any]:
        try:
            with self._span("生成 tool message", "tool", {"step": step, "tool": _tool_name(tool_call)}):
                return self.tool_executor.execute(tool_call, turn_ctx)
        except Exception as exc:
            name = _tool_name(tool_call)
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "name": name,
                "content": json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
            }

    def _span(self, name: str, phase: str, metadata: dict[str, Any] | None = None) -> Any:
        if self.telemetry is None:
            return _NullSpan()
        return self.telemetry.span(name, function="AgentLoop.run", phase=phase, metadata=metadata)


_GENERATED_PROMPT_BLOCK_PATTERNS = (
    (
        re.compile(r"<workspace_context\b[^>]*>.*?</workspace_context>", re.IGNORECASE | re.DOTALL),
        "[generated workspace context omitted]",
    ),
    (
        re.compile(r"<current_task\b[^>]*>.*?</current_task>", re.IGNORECASE | re.DOTALL),
        "[generated current task omitted]",
    ),
    (
        re.compile(r"<coding_agent_prefix\b[^>]*>.*?</coding_agent_prefix>", re.IGNORECASE | re.DOTALL),
        "[generated coding-agent prefix omitted]",
    ),
)


def _sanitize_conversation_message(message: dict[str, Any]) -> dict[str, Any]:
    sanitized = deepcopy(message)
    for field in ("content", "reasoning_content"):
        value = sanitized.get(field)
        if isinstance(value, str):
            sanitized[field] = _sanitize_generated_prompt_text(value)
    return sanitized


def _sanitize_generated_prompt_text(value: str) -> str:
    sanitized = value
    for pattern, replacement in _GENERATED_PROMPT_BLOCK_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    for marker in _GENERATED_PROMPT_MARKERS:
        sanitized = sanitized.replace(marker, marker.replace("<", "&lt;", 1))
    return sanitized


def _tool_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return ""
    name = function.get("name")
    return name if isinstance(name, str) else ""


def _is_parallel_safe_tool(tool_call: dict[str, Any]) -> bool:
    return _tool_name(tool_call) in PARALLEL_SAFE_TOOLS


class _NullSpan:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False
