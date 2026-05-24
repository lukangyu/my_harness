from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from coding_agent.memory.projector import ToolMemoryProjector
from coding_agent.memory.stores.dialog_archive import DialogArchiveStore
from coding_agent.memory.stores.handoff import HandoffStore
from coding_agent.memory.stores.long_term import LongTermMemoryStore
from coding_agent.memory.stores.scratchpad import ScratchpadStore
from coding_agent.memory.stores.tool_result import ToolResultStore


class MemoryStore:
    def __init__(
        self,
        project_root: Path | str,
        dialog_dir: Path | str | None = None,
        tool_result_dir: Path | str | None = None,
        memory_dir: Path | str | None = None,
        conversation_memory_dir: Path | str | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.memory_dir = Path(memory_dir) if memory_dir is not None else self.project_root / ".coding-agent" / "memory"
        self.conversation_memory_dir = (
            Path(conversation_memory_dir)
            if conversation_memory_dir is not None
            else self.project_root / ".coding-agent" / "conversation_memory"
        )
        self.scratchpad_path = self.memory_dir / "scratchpad.json"
        self.handoff_path = self.memory_dir / "handoff.md"
        self.dialog_dir = Path(dialog_dir) if dialog_dir is not None else self.memory_dir / "dialog"
        self.tool_result_dir = Path(tool_result_dir) if tool_result_dir is not None else self.memory_dir / "tool_result"
        self.scratchpad_store = ScratchpadStore(self.memory_dir)
        self.handoff_store = HandoffStore(self.memory_dir)
        self.long_term_store = LongTermMemoryStore(self.conversation_memory_dir)
        self.dialog_archive_store = DialogArchiveStore(self.project_root, self.dialog_dir)
        self.tool_result_store = ToolResultStore(self.project_root, self.tool_result_dir)
        self.projector = ToolMemoryProjector(self, summarize_tool_result)

    def load_scratchpad(self) -> dict[str, Any]:
        return self.scratchpad_store.load()

    def save_scratchpad(self, scratchpad: dict[str, Any]) -> None:
        self.scratchpad_store.save(scratchpad)

    def read_handoff(self) -> str:
        return self.handoff_store.read()

    def write_handoff(self, content: str) -> None:
        self.handoff_store.write(content)

    def archive_dialog_messages(self, messages: list[dict[str, Any]]) -> Path | None:
        return self.dialog_archive_store.archive_messages(messages)

    def offload_tool_result(
        self,
        *,
        tool: str,
        content: str,
        max_inline_chars: int = 4000,
    ) -> dict[str, Any]:
        return self.tool_result_store.offload(
            tool=tool,
            content=content,
            max_inline_chars=max_inline_chars,
        )

    def record_tool_result(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        self.projector.record_tool_result(
            tool=tool,
            arguments=arguments,
            result=result,
        )

    def append_long_term_memories(
        self,
        memories: list[dict[str, Any]],
        *,
        source: str,
        evidence: list[str],
    ) -> list[dict[str, Any]]:
        return self.long_term_store.append_memories(memories, source=source, evidence=evidence)

    def render_memory_anchor(self, *, max_chars: int) -> str:
        scratchpad = self.load_scratchpad()
        text = "<memory_anchor>\n" + json.dumps(scratchpad, indent=2, ensure_ascii=False) + "\n</memory_anchor>"
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... [memory_anchor truncated]\n</memory_anchor>"

    def _resolve_project_path(self, path: str | Path) -> Path:
        raw = Path(path)
        if raw.is_absolute():
            return raw.resolve()
        return (self.project_root / raw).resolve()

    def _relative_project_path(self, path: str | Path) -> str:
        absolute = self._resolve_project_path(path)
        try:
            return absolute.relative_to(self.project_root.resolve()).as_posix()
        except ValueError:
            return Path(path).as_posix()


def summarize_tool_result(tool: str, arguments: dict[str, Any], result: dict[str, Any]) -> str:
    if result.get("ok") is False:
        return f"{tool} 失败：{result.get('error') or result.get('stderr') or '未知错误'}"
    if tool == "read_file":
        return f"读取文件 {result.get('path')}，返回 {len(result.get('content') or '')} 字符"
    if tool == "write_file":
        return f"写入文件 {result.get('path')}"
    if tool == "apply_patch":
        changed = result.get("changed_files") or []
        return f"应用 patch，变更 {len(changed)} 个文件：{changed[:5]}"
    if tool == "search_text":
        matches = result.get("matches") or []
        return f"搜索 {arguments.get('query')!r}，命中 {len(matches)} 条"
    if tool == "run_shell":
        return f"执行命令 {result.get('command')!r}，exit_code={result.get('exit_code')}"
    return f"{tool} 执行完成"
