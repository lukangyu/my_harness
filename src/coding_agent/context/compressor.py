from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coding_agent.context.context import CompactClient, Context, build_compaction_prompt
from coding_agent.memory.store import MemoryStore


@dataclass(frozen=True)
class CompressionResult:
    compacted: bool
    summary_message: dict[str, Any] | None = None
    remaining_messages: list[dict[str, Any]] | None = None
    archive_path: str | None = None
    summary: str = ""
    memories: list[dict[str, Any]] | None = None


class ContextCompressor:
    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        compact_client: CompactClient,
    ) -> None:
        self.memory_store = memory_store
        self.compact_client = compact_client

    def should_compress(self, context: Context) -> bool:
        threshold = self.threshold_tokens(context)
        return context.estimate_active_tokens() > threshold

    def threshold_tokens(self, context: Context) -> int:
        return int(context.options.max_input_tokens * context.options.compact_threshold_ratio)

    def compress(self, context: Context) -> CompressionResult:
        old_messages, remaining_messages = context.slice_old_history(
            tail_token_budget=self._tail_token_budget(context),
        )
        if not old_messages:
            return CompressionResult(compacted=False)

        archive_path = self.memory_store.archive_dialog_messages(old_messages)
        source_refs = self._source_refs(archive_path)
        compact_prompt = build_compaction_prompt(
            old_messages=old_messages,
            previous_handoff=self.memory_store.read_handoff(),
            scratchpad=self.memory_store.load_scratchpad(),
            source_refs=source_refs,
        )
        response = self.compact_client.chat(
            [
                {"role": "system", "content": "你是上下文压缩器，只输出交接备忘录。"},
                {"role": "user", "content": compact_prompt},
            ],
            [],
        )
        compaction_payload = parse_compaction_response(_content_from_response(response))
        summary = compaction_payload.handoff
        if not summary:
            summary = "旧对话已归档，必要时读取 archive_log_path 指向的 JSONL。"
        self.memory_store.write_handoff(summary)
        memories = self.memory_store.append_long_term_memories(
            compaction_payload.memories,
            source="context_compaction",
            evidence=[path for path in [source_refs["dialog_path"]] if path],
        )
        summary_payload = {
            "summary": summary,
            "archive_log_path": source_refs["dialog_path"],
            "instruction": (
                "Past conversation history has been archived. "
                "Use read_file on archive_log_path if cross-session facts are missing."
            ),
        }
        context.replace_active_history(summary_payload, remaining_messages)
        summary_message = context.history_frames()[0].payload if context.history_frames() else None
        return CompressionResult(
            compacted=True,
            summary_message=summary_message,
            remaining_messages=remaining_messages,
            archive_path=source_refs["dialog_path"],
            summary=summary,
            memories=memories,
        )

    def _tail_token_budget(self, context: Context) -> int:
        return max(1, int(self.threshold_tokens(context) * context.options.compact_tail_ratio))

    def _source_refs(self, archive_path: Path | None) -> dict[str, str | None]:
        return {
            "dialog_path": _relative_or_none(archive_path, self.memory_store.project_root),
            "tool_result_dir": self.memory_store.tool_result_dir.relative_to(
                self.memory_store.project_root
            ).as_posix(),
        }


@dataclass(frozen=True)
class CompactionPayload:
    handoff: str
    memories: list[dict[str, Any]]


def parse_compaction_response(text: str) -> CompactionPayload:
    stripped = text.strip()
    if not stripped:
        return CompactionPayload(handoff="", memories=[])
    parsed = _load_json_object(stripped)
    if parsed is None:
        return CompactionPayload(handoff=stripped, memories=[])
    handoff = parsed.get("handoff")
    memories = parsed.get("memories")
    if not isinstance(handoff, str):
        return CompactionPayload(handoff=stripped, memories=[])
    return CompactionPayload(handoff=handoff.strip(), memories=_valid_memory_candidates(memories))


def _load_json_object(text: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            return None
        try:
            loaded = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return loaded if isinstance(loaded, dict) else None


def _valid_memory_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    candidates: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        memory_type = item.get("type")
        content = item.get("content")
        reason = item.get("reason")
        confidence = item.get("confidence")
        if memory_type not in {"personal", "procedural", "knowledge"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        if not isinstance(reason, str) or not reason.strip():
            continue
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            continue
        candidates.append(
            {
                "type": memory_type,
                "content": content.strip(),
                "confidence": max(0.0, min(1.0, float(confidence))),
                "reason": reason.strip(),
            }
        )
    return candidates


def _content_from_response(response: dict[str, Any]) -> str:
    message = response.get("message", {})
    content = message.get("content") if isinstance(message, dict) else ""
    return content.strip() if isinstance(content, str) else ""


def _relative_or_none(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    return path.relative_to(root).as_posix()
