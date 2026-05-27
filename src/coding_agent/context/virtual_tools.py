from __future__ import annotations

import json
from typing import Any


SYNC_SESSION_CONTEXT = "sync_session_context"
SYNC_COMPACTED_CONTEXT = "sync_compacted_context"
AUTO_MEMORY_SEARCH = "auto_memory_search"


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


def _xml_json(tag: str, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"<{tag}>\n{text}\n</{tag}>"
