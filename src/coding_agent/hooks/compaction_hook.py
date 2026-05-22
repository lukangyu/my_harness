from __future__ import annotations

from pathlib import Path
from typing import Any

from coding_agent.context.context import (
    CompactClient,
    Context,
    build_handoff_prompt,
    partition_messages,
)
from coding_agent.memory.store import MemoryStore
from coding_agent.orchestrator.lifecycle import AgentLifecycleHook, AgentTurnContext


class ContextCompactionHook(AgentLifecycleHook):
    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        compact_client: CompactClient,
    ) -> None:
        self.memory_store = memory_store
        self.compact_client = compact_client

    def pre_llm(self, ctx: AgentTurnContext) -> None:
        context = ctx.context_entity
        if context is None:
            return
        threshold = int(context.options.max_input_tokens * context.options.compact_threshold_ratio)
        if context.estimate_active_tokens() <= threshold:
            return
        self._compact_and_archive_dialog(context)

    def _compact_and_archive_dialog(self, context: Context) -> None:
        old_messages, remaining_messages = context.slice_old_history()
        if not old_messages:
            return

        archive_path = self.memory_store.archive_dialog_messages(old_messages)
        source_refs = {
            "dialog_path": _relative_or_none(archive_path, self.memory_store.project_root),
            "tool_index_path": self.memory_store.tool_index_path.relative_to(
                self.memory_store.project_root
            ).as_posix(),
            "tool_result_dir": self.memory_store.tool_result_dir.relative_to(
                self.memory_store.project_root
            ).as_posix(),
        }
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
        message = response.get("message", {})
        summary = message.get("content") if isinstance(message, dict) else ""
        if not isinstance(summary, str) or not summary.strip():
            summary = "旧对话已归档，必要时读取 archive_log_path 指向的 JSONL。"
        summary_payload = {
            "summary": summary.strip(),
            "archive_log_path": source_refs["dialog_path"],
            "instruction": (
                "Past conversation history has been archived. "
                "Use read_file on archive_log_path if cross-session facts are missing."
            ),
        }
        context.replace_active_history(summary_payload, remaining_messages)


def _relative_or_none(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    return path.relative_to(root).as_posix()
