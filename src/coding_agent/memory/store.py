from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from coding_agent.memory.projector import ToolMemoryProjector
from coding_agent.memory.stores.dialog_archive import DialogArchiveStore
from coding_agent.memory.stores.file_summary import FileSummaryStore, is_summary_valid
from coding_agent.memory.stores.handoff import HandoffStore
from coding_agent.memory.stores.scratchpad import ScratchpadStore
from coding_agent.memory.stores.tool_index import ToolIndexStore
from coding_agent.memory.stores.tool_result import ToolResultStore


class MemoryStore:
    def __init__(
        self,
        project_root: Path | str,
        dialog_dir: Path | str | None = None,
        tool_result_dir: Path | str | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.memory_dir = self.project_root / ".coding-agent" / "memory"
        self.scratchpad_path = self.memory_dir / "scratchpad.json"
        self.handoff_path = self.memory_dir / "handoff.md"
        self.tool_index_path = self.memory_dir / "tool_index.jsonl"
        self.file_summaries_path = self.memory_dir / "file_summaries.json"
        self.dialog_dir = Path(dialog_dir) if dialog_dir is not None else self.memory_dir / "dialog"
        self.tool_result_dir = Path(tool_result_dir) if tool_result_dir is not None else self.memory_dir / "tool_result"
        self.scratchpad_store = ScratchpadStore(self.memory_dir)
        self.handoff_store = HandoffStore(self.memory_dir)
        self.tool_index_store = ToolIndexStore(self.memory_dir)
        self.file_summary_store = FileSummaryStore(self.project_root, self.memory_dir)
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

    def append_tool_index(self, record: dict[str, Any]) -> None:
        self.tool_index_store.append(record)

    def load_file_summaries(self) -> dict[str, dict[str, Any]]:
        return self.file_summary_store.load()

    def save_file_summaries(self, summaries: dict[str, dict[str, Any]]) -> None:
        self.file_summary_store.save(summaries)

    def update_file_summary(self, path: str | Path) -> dict[str, Any] | None:
        return self.file_summary_store.update(path)

    def invalidate_file_summary(self, path: str | Path, reason: str) -> None:
        self.file_summary_store.invalidate(path, reason)

    def render_file_summaries(
        self,
        *,
        candidate_paths: list[str],
        max_count: int,
        max_chars: int,
    ) -> str:
        summaries = self.load_file_summaries()
        valid: list[dict[str, Any]] = []
        changed = False
        for path in sorted(dict.fromkeys(candidate_paths)):
            summary = summaries.get(path)
            if summary is None:
                continue
            absolute = self._resolve_project_path(path)
            if not is_summary_valid(summary, absolute):
                summary["stale"] = True
                summary["stale_reason"] = "hash_mismatch_or_missing"
                summaries[path] = summary
                changed = True
                continue
            valid.append(summary)
            if len(valid) >= max_count:
                break
        if changed:
            self.save_file_summaries(summaries)
        if not valid:
            return ""
        rendered = _render_file_summaries(valid)
        if len(rendered) <= max_chars:
            return rendered
        return rendered[:max_chars] + "\n... [file_summaries truncated]\n</file_summaries>"

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

    def candidate_summary_paths(self) -> list[str]:
        scratchpad = self.load_scratchpad()
        paths: list[str] = []
        for key in ("read_files", "modified_files"):
            values = scratchpad.get(key)
            if isinstance(values, list):
                paths.extend(value for value in values if isinstance(value, str))
        return paths

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


def _render_file_summaries(summaries: list[dict[str, Any]]) -> str:
    lines = ["<file_summaries>"]
    for summary in summaries:
        lines.append(f'<file path="{summary.get("path", "")}" hash="{summary.get("content_hash", "")}" language="{summary.get("language", "")}">')
        imports = summary.get("imports") or []
        lines.append("imports:")
        lines.extend(f"- {item}" for item in imports[:30])
        if not imports:
            lines.append("- none")
        lines.append("symbols:")
        symbols = summary.get("symbols") or []
        if not symbols:
            lines.append("- none")
        for symbol in symbols[:30]:
            if not isinstance(symbol, dict):
                continue
            lines.append(f'- {symbol.get("type")} {symbol.get("name")} line {symbol.get("line")}')
            docstring = symbol.get("docstring")
            if isinstance(docstring, str) and docstring:
                lines.append(f"  docstring: {docstring}")
            preview = symbol.get("preview") or []
            if preview:
                lines.append("  preview:")
                lines.extend(f"    {line}" for line in preview[:8])
        lines.append("</file>")
    lines.append("</file_summaries>")
    return "\n".join(lines)

