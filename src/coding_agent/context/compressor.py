from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coding_agent.context.context import CompactClient, Context, build_handoff_prompt
from coding_agent.memory.store import MemoryStore


@dataclass(frozen=True)
class CompressionResult:
    compacted: bool
    summary_message: dict[str, Any] | None = None
    remaining_messages: list[dict[str, Any]] | None = None
    archive_path: str | None = None
    summary: str = ""


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
        compact_prompt = build_handoff_prompt(
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
        summary = _summary_from_response(response)
        if not summary:
            summary = "旧对话已归档，必要时读取 archive_log_path 指向的 JSONL。"
        self.memory_store.write_handoff(summary)
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


def _summary_from_response(response: dict[str, Any]) -> str:
    message = response.get("message", {})
    summary = message.get("content") if isinstance(message, dict) else ""
    return summary.strip() if isinstance(summary, str) else ""


def _relative_or_none(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    return path.relative_to(root).as_posix()
