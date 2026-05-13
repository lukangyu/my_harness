from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from coding_agent.context import ContextManager, PromptBuilder, WorkspaceContextOptions
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


class AgentLoop:
    def __init__(
        self,
        client: ChatClient,
        tools: ToolRegistry,
        max_steps: int,
        cwd: Path,
        context_options: WorkspaceContextOptions | None = None,
        recent_message_tokens: int = 12000,
    ) -> None:
        self.client = client
        self.tools = tools
        self.max_steps = max_steps
        self.context_manager = ContextManager(
            cwd=cwd,
            options=context_options or WorkspaceContextOptions(),
            recent_message_tokens=recent_message_tokens,
        )
        self.prompt_builder = PromptBuilder()

    def run(
        self,
        task: str,
        prior_messages: list[dict[str, Any]] | None = None,
        mode: str = "run",
    ) -> AgentResult:
        tool_schemas = self.tools.schemas()
        envelope = self.context_manager.build(
            task=task,
            prior_messages=list(prior_messages or []),
            tool_schemas=tool_schemas,
            mode=mode,
        )
        messages = self.prompt_builder.to_messages(envelope, mode=mode)
        conversation_messages = deepcopy(envelope.recent_messages)
        conversation_messages.append({"role": "user", "content": envelope.current_task})

        for _ in range(self.max_steps):
            response = self.client.chat(messages, tool_schemas)
            assistant_message = response["message"]
            messages.append(assistant_message)
            conversation_messages.append(deepcopy(assistant_message))

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                return AgentResult(
                    final_answer=assistant_message.get("content") or "",
                    messages=messages,
                    conversation_messages=conversation_messages,
                )

            for tool_call in tool_calls:
                tool_message = self._tool_message(tool_call)
                messages.append(tool_message)
                conversation_messages.append(deepcopy(tool_message))

        return AgentResult(
            final_answer="Stopped after reaching max steps.",
            messages=messages,
            conversation_messages=conversation_messages,
            reached_max_steps=True,
        )

    def _tool_message(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function", {})
        name = function.get("name", "")
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
            result = self.tools.call(name, arguments)
        except (json.JSONDecodeError, ValueError) as exc:
            result = {"ok": False, "error": f"Invalid JSON arguments: {exc}"}

        return {
            "role": "tool",
            "tool_call_id": tool_call.get("id", ""),
            "name": name,
            "content": json.dumps(result, ensure_ascii=False),
        }
