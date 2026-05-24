from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from coding_agent.context.context import Context, ContextFrame
from coding_agent.context.virtual_tools import (
    SYNC_COMPACTED_CONTEXT,
    SYNC_SESSION_CONTEXT,
    synthetic_tool_pair,
)


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
        ]
        messages.extend(self._session_context_messages(context))
        messages.extend(self._compacted_context_messages(context, active_frames))
        messages.extend(frame.payload for frame in active_frames if frame.role is not None and frame.kind != "compact_summary")
        messages.append({"role": "user", "content": context.task})
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
            "`agent_context/tool_result/xxxx.txt` or `agent_context/dialog/xxxx.jsonl` in context frames."
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

    def _session_context_messages(self, context: Context) -> list[dict[str, Any]]:
        return synthetic_tool_pair(
            name=SYNC_SESSION_CONTEXT,
            call_id="call_sync_session_context",
            tag="session_context",
            payload=self._session_context_payload(context),
        )

    def _compacted_context_messages(
        self,
        context: Context,
        active_frames: list[ContextFrame],
    ) -> list[dict[str, Any]]:
        compact_frames = [frame for frame in active_frames if frame.kind == "compact_summary"]
        if not compact_frames and not context.handoff:
            return []
        payload = {
            "kind": "compacted_context",
            "schema_version": 1,
            "handoff": context.handoff,
            "compact_summary": "\n\n".join(
                str(frame.payload.get("content") or "").strip()
                for frame in compact_frames
                if str(frame.payload.get("content") or "").strip()
            ),
            "archive_paths": self._archive_paths(compact_frames),
            "carried_scratchpad": context.scratchpad,
        }
        return synthetic_tool_pair(
            name=SYNC_COMPACTED_CONTEXT,
            call_id="call_sync_compacted_context",
            tag="compacted_context",
            payload=payload,
        )

    def _session_context_payload(self, context: Context) -> dict[str, Any]:
        snapshot = context.workspace_snapshot
        return {
            "kind": "session_context",
            "schema_version": 1,
            "cwd": str(snapshot.cwd),
            "workspace_root": str(snapshot.repo_root or snapshot.cwd),
            "initial_git_branch": snapshot.branch,
            "agent_context": {
                "dialog": "agent_context/dialog/",
                "tool_result": "agent_context/tool_result/",
            },
            "notes": [
                "Tool definitions are provided through the API tools array.",
                "Archived paths can be inspected with read_file when exact details are needed.",
            ],
        }

    def _archive_paths(self, frames: list[ContextFrame]) -> list[dict[str, str]]:
        paths: list[dict[str, str]] = []
        for frame in frames:
            content = str(frame.payload.get("content") or "")
            for token in content.replace("\n", " ").split():
                cleaned = token.strip("`'\".,;:()[]")
                if "agent_context/dialog/" in cleaned or cleaned.endswith(".jsonl"):
                    paths.append(
                        {
                            "kind": "dialog",
                            "path": cleaned,
                            "instruction": "Use read_file when exact old dialog details are needed.",
                        }
                    )
        return paths

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
            "Internal protocol tools are not user-facing capabilities; do not list them as available tools. "
            "Only call memory or session_search when that tool exists in the current tool schema."
        )
        .add_tool_enforcement(
            "Work only through provided tools. File access is limited by the workspace sandbox. "
            "Shell commands are subject to command policy. If a tool is rejected or fails, "
            "report the reason and adapt."
        )
    )
