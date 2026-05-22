from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from coding_agent.context.context import Context, ContextFrame


ContextFormat = Literal["json", "xml"]


@dataclass(frozen=True)
class PromptLayer:
    name: str
    content: str


@dataclass(frozen=True)
class PromptState:
    layers: tuple[PromptLayer, ...]
    context_format: ContextFormat


class PromptBuilder:
    def __init__(self, *, context_format: ContextFormat = "json") -> None:
        self._state = PromptState(layers=(), context_format=context_format)

    @property
    def state(self) -> PromptState:
        return self._state

    def add_system_base(self, content: str) -> "PromptBuilder":
        return self._add_or_replace("system_base", content)

    def add_tool_guidance(self, content: str) -> "PromptBuilder":
        return self._add_or_replace("tool_guidance", content)

    def add_tool_enforcement(self, content: str) -> "PromptBuilder":
        return self._add_or_replace("tool_enforcement", content)

    def modify_system_prompt(self, layer_name: str, content: str) -> None:
        layers = list(self._state.layers)
        for index, layer in enumerate(layers):
            if layer.name == layer_name:
                layers[index] = PromptLayer(layer_name, content)
                self._state = PromptState(
                    layers=tuple(layers),
                    context_format=self._state.context_format,
                )
                return
        raise KeyError(layer_name)

    def build_final_messages(self, context: Context) -> list[dict[str, Any]]:
        active_frames = context.history_frames()
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(context),
            },
            {
                "role": "user",
                "content": self._wrap_context(
                    [
                        *context.context_frames(),
                        *[frame for frame in active_frames if frame.role is None],
                    ]
                ),
            },
        ]
        messages.extend(frame.payload for frame in active_frames if frame.role is not None)
        messages.append(
            {
                "role": "user",
                "content": self._wrap_task(context.task_frame()),
            }
        )
        return deepcopy(messages)

    def _add_or_replace(self, name: str, content: str) -> "PromptBuilder":
        layers = [layer for layer in self._state.layers if layer.name != name]
        layers.append(PromptLayer(name=name, content=content))
        order = {"system_base": 0, "tool_guidance": 1, "tool_enforcement": 2}
        layers.sort(key=lambda layer: order.get(layer.name, 99))
        self._state = PromptState(layers=tuple(layers), context_format=self._state.context_format)
        return self

    def _system_prompt(self, context: Context) -> str:
        parts = [layer.content for layer in self._state.layers if layer.content]
        availability_notes = self._tool_availability_notes(context)
        if availability_notes:
            parts.append(availability_notes)
        parts.append(self._context_management_notice(context))
        return "\n\n".join(parts)

    def _tool_availability_notes(self, context: Context) -> str:
        notes: list[str] = []
        if not self._has_tool(context, "memory"):
            notes.append("Note: memory tool is currently offline. Do not attempt to invoke it.")
        if not self._has_tool(context, "session_search"):
            notes.append("Note: session_search tool is currently offline. Do not attempt to invoke it.")
        return "\n".join(notes)

    def _has_tool(self, context: Context, name: str) -> bool:
        for schema in context.tool_schemas:
            if not isinstance(schema, dict):
                continue
            function = schema.get("function")
            if isinstance(function, dict) and function.get("name") == name:
                return True
        return False

    def _context_management_notice(self, context: Context) -> str:
        base = (
            "CONTEXT MANAGEMENT NOTICE:\n"
            "To fit into the context window, massive tool outputs and archived dialogs "
            "may be offloaded to local files. You may see reference paths like "
            "`tool_result/xxxx.txt` or `dialog/xxxx.jsonl` in context frames."
        )
        if self._has_tool(context, "read_file"):
            return (
                base
                + "\nIf historical details, code execution logs, or specific past user "
                "instructions are needed, do not ask the user to repeat recoverable "
                "information; use your read_file tool to inspect those paths."
            )
        return (
            base
            + "\nTreat those paths as archive references. Do not attempt to call tools "
            "that are absent from the current tool schema."
        )

    def _wrap_context(self, frames: list[ContextFrame]) -> str:
        payload = {
            "kind": "context",
            "frames": [
                {
                    "kind": frame.kind,
                    "payload": frame.payload,
                    "priority": frame.priority,
                    "stability": frame.stability,
                    "token_estimate": frame.token_estimate,
                }
                for frame in frames
            ],
        }
        return self._serialize(payload, "context_json")

    def _wrap_task(self, frame: ContextFrame) -> str:
        payload = {
            "kind": "current_task",
            "mode": frame.payload["mode"],
            "content": frame.payload["content"],
        }
        return self._serialize(payload, "current_task_json")

    def _serialize(self, payload: dict[str, Any], xml_tag: str) -> str:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if self._state.context_format == "xml":
            return f"<{xml_tag}>\n{text}\n</{xml_tag}>"
        return text


def create_default_prompt_builder() -> PromptBuilder:
    return (
        PromptBuilder()
        .add_system_base(
            "You are coding-agent, a local AI coding assistant.\n"
            "- Inspect relevant files before editing.\n"
            "- Prefer small, focused changes.\n"
            "- Verify changes with allowed commands when practical.\n"
            "- Do not claim tests passed unless tool results show they passed."
        )
        .add_tool_guidance(
            "Use the provided tool schemas as the sole source of available tools. "
            "When asked what tools are available, answer only from the current tools array. "
            "Do not include host, developer, orchestration, or non-schema capabilities as tools. "
            "Only call memory or session_search when that tool exists in the current tool schema."
        )
        .add_tool_enforcement(
            "Work only through provided tools. File access is limited by the workspace sandbox. "
            "Shell commands are subject to command policy. If a tool is rejected or fails, "
            "report the reason and adapt."
        )
    )
