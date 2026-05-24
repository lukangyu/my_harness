from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


SYNC_SESSION_CONTEXT = "sync_session_context"
SYNC_COMPACTED_CONTEXT = "sync_compacted_context"
AUTO_MEMORY_SEARCH = "auto_memory_search"
NOTIFY_CONTEXT_INVALIDATED = "notify_context_invalidated"

INTERNAL_TOOL_NAMES = {
    SYNC_SESSION_CONTEXT,
    SYNC_COMPACTED_CONTEXT,
    AUTO_MEMORY_SEARCH,
    NOTIFY_CONTEXT_INVALIDATED,
}


def internal_tool_schemas() -> list[dict[str, Any]]:
    return [
        _schema(
            SYNC_SESSION_CONTEXT,
            "Internal protocol tool used to inject stable session context.",
        ),
        _schema(
            SYNC_COMPACTED_CONTEXT,
            "Internal protocol tool used to inject compacted session handoff context.",
        ),
        _schema(
            AUTO_MEMORY_SEARCH,
            "Internal protocol tool used to inject automatic long-term memory search results.",
            properties={"query": {"type": "string"}},
        ),
        _schema(
            NOTIFY_CONTEXT_INVALIDATED,
            "Internal protocol tool used to notify the model that session context is stale.",
            properties={"reason": {"type": "string"}},
        ),
    ]


def synthetic_tool_pair(
    *,
    name: str,
    call_id: str,
    arguments: dict[str, Any] | None = None,
    tag: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    args = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
    content = _xml_json(tag, payload)
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        },
    ]


def is_internal_tool(name: str) -> bool:
    return name in INTERNAL_TOOL_NAMES


def handle_internal_tool(name: str, arguments: dict[str, Any], ctx: Any) -> dict[str, Any]:
    if name == AUTO_MEMORY_SEARCH:
        return {
            "ok": True,
            "internal": True,
            "kind": "memory_search_results",
            "query": arguments.get("query", ""),
            "results": [],
        }
    if name == NOTIFY_CONTEXT_INVALIDATED:
        return {
            "ok": True,
            "internal": True,
            "kind": "context_invalidated",
            "reason": arguments.get("reason", "unspecified"),
        }
    if name == SYNC_SESSION_CONTEXT:
        context = getattr(ctx, "context_entity", None)
        payload = _session_context_payload(context) if context is not None else {}
        return {"ok": True, "internal": True, **payload}
    if name == SYNC_COMPACTED_CONTEXT:
        context = getattr(ctx, "context_entity", None)
        payload = _compacted_context_payload(context) if context is not None else {}
        return {"ok": True, "internal": True, **payload}
    return {"ok": False, "error": f"Unknown internal tool: {name}"}


def _schema(
    name: str,
    description: str,
    *,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": deepcopy(properties or {}),
                "additionalProperties": False,
            },
        },
    }


def _xml_json(tag: str, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"<{tag}>\n{text}\n</{tag}>"


def _session_context_payload(context: Any) -> dict[str, Any]:
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


def _compacted_context_payload(context: Any) -> dict[str, Any]:
    archive_paths = []
    handoff = ""
    compact_summary = ""
    carried_scratchpad: dict[str, Any] = {}
    for frame in context.history_frames():
        if frame.kind != "compact_summary":
            continue
        content = frame.payload.get("content")
        if isinstance(content, str):
            compact_summary = content
    if isinstance(context.handoff, str) and context.handoff.strip():
        handoff = context.handoff.strip()
    if isinstance(context.scratchpad, dict):
        carried_scratchpad = deepcopy(context.scratchpad)
    if compact_summary:
        archive_paths = _archive_paths_from_text(compact_summary)
    return {
        "kind": "compacted_context",
        "schema_version": 1,
        "handoff": handoff,
        "compact_summary": compact_summary,
        "archive_paths": archive_paths,
        "carried_scratchpad": carried_scratchpad,
    }


def _archive_paths_from_text(text: str) -> list[dict[str, str]]:
    paths: list[dict[str, str]] = []
    for token in text.replace("\n", " ").split():
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
