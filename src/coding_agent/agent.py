from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from coding_agent.tools import ToolRegistry


SYSTEM_PROMPT = "You are a coding agent. Use tools when needed and return a concise final answer."


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        ...


@dataclass
class AgentResult:
    final_answer: str
    messages: list[dict[str, Any]]
    reached_max_steps: bool = False


class AgentLoop:
    def __init__(self, client: ChatClient, tools: ToolRegistry, max_steps: int) -> None:
        self.client = client
        self.tools = tools
        self.max_steps = max_steps

    def run(
        self,
        task: str,
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> AgentResult:
        messages = list(prior_messages or [])
        if not messages:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": task})

        for _ in range(self.max_steps):
            response = self.client.chat(messages, self.tools.schemas())
            assistant_message = response["message"]
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                return AgentResult(final_answer=assistant_message.get("content") or "", messages=messages)

            for tool_call in tool_calls:
                messages.append(self._tool_message(tool_call))

        return AgentResult(
            final_answer="Stopped after reaching max steps.",
            messages=messages,
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
