from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from coding_agent.context import (
    ContextManager,
    PromptBuilder,
    UsageStats,
    WorkspaceContextOptions,
)
from coding_agent.memory import MemoryStore
from coding_agent.telemetry import TelemetryLogger
from coding_agent.tools import ToolRegistry


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
        telemetry: TelemetryLogger | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.client = client
        self.tools = tools
        self.max_steps = max_steps
        self.on_tool_call = on_tool_call
        self.telemetry = telemetry
        self.memory_store = memory_store
        self.context_manager = ContextManager(
            cwd=cwd,
            options=context_options or WorkspaceContextOptions(),
            recent_message_tokens=recent_message_tokens,
            memory_store=memory_store,
            compact_client=client,
        )
        self.prompt_builder = PromptBuilder()

    def run(
        self,
        task: str,
        prior_messages: list[dict[str, Any]] | None = None,
        mode: str = "run",
    ) -> AgentResult:
        self._event("agent.run.start", "AgentLoop 开始执行任务", "run", {"mode": mode, "task_length": len(task)})
        tool_schemas = self.tools.schemas()
        with self._span("构建上下文", "context", {"tool_count": len(tool_schemas)}):
            envelope = self.context_manager.build(
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
                "full_context_key": envelope.full_context_key,
                "recent_messages": len(envelope.recent_messages),
                "workspace_fingerprint": envelope.workspace_prefix.workspace_fingerprint,
            },
        )
        with self._span("渲染 prompt messages", "prompt"):
            messages = self.prompt_builder.to_messages(envelope, mode=mode)
        conversation_messages = deepcopy(envelope.recent_messages)
        conversation_messages.append({"role": "user", "content": envelope.current_task})
        usage: UsageStats | None = None

        for step in range(1, self.max_steps + 1):
            self._event(
                "agent.step.start",
                f"开始第 {step} 轮 AgentLoop",
                "agent_step",
                {"step": step, "message_count": len(messages)},
            )
            with self._span("调用大模型", "llm", {"step": step, "message_count": len(messages)}):
                response = self.client.chat(messages, tool_schemas)
            response_usage = response.get("usage")
            if response_usage is not None:
                usage = response_usage
            assistant_message = response["message"]
            messages.append(assistant_message)
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
                )

            for tool_call in tool_calls:
                with self._span("生成 tool message", "tool", {"step": step}):
                    tool_message = self._tool_message(tool_call)
                messages.append(tool_message)
                conversation_messages.append(deepcopy(tool_message))

        self._event("agent.max_steps", "AgentLoop 达到最大步数后停止", "run", {"max_steps": self.max_steps})
        return AgentResult(
            final_answer="Stopped after reaching max steps.",
            messages=messages,
            conversation_messages=conversation_messages,
            reached_max_steps=True,
            usage=usage,
        )

    def _tool_message(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        if self.on_tool_call is not None:
            self.on_tool_call(tool_call)
        function = tool_call.get("function", {})
        name = function.get("name", "")
        raw_arguments = function.get("arguments") or "{}"
        if self.telemetry is not None:
            self.telemetry.event(
                "agent.tool_message.start",
                f"开始处理模型请求的工具调用 {name}",
                function="AgentLoop._tool_message",
                phase="tool",
                metadata={"tool": name, "arguments_length": len(raw_arguments)},
            )
        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            result = self.tools.call(name, arguments)
        except (json.JSONDecodeError, ValueError) as exc:
            result = {"ok": False, "error": f"Invalid JSON arguments: {exc}"}
        if self.telemetry is not None:
            self.telemetry.event(
                "agent.tool_message.end",
                f"工具调用 {name} 的 tool message 已生成",
                function="AgentLoop._tool_message",
                phase="tool",
                metadata={"tool": name, "ok": result.get("ok"), "content_length": len(json.dumps(result, ensure_ascii=False))},
            )

        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "name": name,
            "content": json.dumps(result, ensure_ascii=False),
        }

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

    def _span(self, name: str, phase: str, metadata: dict[str, Any] | None = None) -> Any:
        if self.telemetry is None:
            return _NullSpan()
        return self.telemetry.span(name, function="AgentLoop.run", phase=phase, metadata=metadata)


class _NullSpan:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False
