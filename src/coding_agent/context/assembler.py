from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coding_agent.context.context import Context, WorkspaceContextOptions, WorkspaceSnapshot
from coding_agent.memory.store import MemoryStore


class ContextAssembler:
    def __init__(
        self,
        *,
        cwd: Path,
        options: WorkspaceContextOptions,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.cwd = cwd
        self.options = options
        self.memory_store = memory_store

    def build(
        self,
        *,
        task: str,
        prior_messages: list[dict[str, Any]] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        mode: str = "run",
    ) -> Context:
        return Context(
            cwd=self.cwd,
            options=self.options,
            task=task,
            prior_messages=prior_messages,
            tool_schemas=tool_schemas,
            mode=mode,
            workspace_snapshot=WorkspaceSnapshot.build(self.cwd, self.options),
            scratchpad=self._read_scratchpad(),
            handoff=self._read_handoff(),
        )

    def _read_scratchpad(self) -> dict[str, Any]:
        if self.memory_store is None:
            return {}
        scratchpad = self.memory_store.load_scratchpad()
        text = json.dumps(scratchpad, ensure_ascii=False, sort_keys=True)
        if len(text) <= self.options.scratchpad_max_chars:
            return scratchpad
        return {
            "_truncated": True,
            "json_prefix": text[: self.options.scratchpad_max_chars],
        }

    def _read_handoff(self) -> str:
        if self.memory_store is None:
            return ""
        return self.memory_store.read_handoff().strip()
