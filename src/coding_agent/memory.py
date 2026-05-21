from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from coding_agent.file_summary import file_hash, is_summary_valid, summarize_file


DEFAULT_SCRATCHPAD: dict[str, Any] = {
    "project_goal": "",
    "user_preferences": [],
    "confirmed_decisions": [],
    "modified_files": [],
    "read_files": [],
    "known_issues": [],
    "active_todos": [],
    "last_verified_commands": [],
}


class MemoryStore:
    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root)
        self.memory_dir = self.project_root / ".coding-agent" / "memory"
        self.scratchpad_path = self.memory_dir / "scratchpad.json"
        self.handoff_path = self.memory_dir / "handoff.md"
        self.tool_index_path = self.memory_dir / "tool_index.jsonl"
        self.file_summaries_path = self.memory_dir / "file_summaries.json"

    def load_scratchpad(self) -> dict[str, Any]:
        if not self.scratchpad_path.exists():
            return dict(DEFAULT_SCRATCHPAD)
        try:
            data = json.loads(self.scratchpad_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return dict(DEFAULT_SCRATCHPAD)
        if not isinstance(data, dict):
            return dict(DEFAULT_SCRATCHPAD)
        scratchpad = dict(DEFAULT_SCRATCHPAD)
        scratchpad.update(data)
        return scratchpad

    def save_scratchpad(self, scratchpad: dict[str, Any]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.scratchpad_path.write_text(
            json.dumps(_normalize_scratchpad(scratchpad), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def read_handoff(self) -> str:
        if not self.handoff_path.exists():
            return ""
        try:
            return self.handoff_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def write_handoff(self, content: str) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.handoff_path.write_text(content.strip() + "\n", encoding="utf-8")

    def append_tool_index(self, record: dict[str, Any]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        enriched = {"timestamp": _now(), **record}
        with self.tool_index_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(enriched, ensure_ascii=False, default=str) + "\n")

    def load_file_summaries(self) -> dict[str, dict[str, Any]]:
        if not self.file_summaries_path.exists():
            return {}
        try:
            data = json.loads(self.file_summaries_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {path: value for path, value in data.items() if isinstance(path, str) and isinstance(value, dict)}

    def save_file_summaries(self, summaries: dict[str, dict[str, Any]]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.file_summaries_path.write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def update_file_summary(self, path: str | Path) -> dict[str, Any] | None:
        absolute = self._resolve_project_path(path)
        if not absolute.is_file():
            return None
        summary = summarize_file(absolute, self.project_root.resolve()).to_dict()
        summaries = self.load_file_summaries()
        summaries[summary["path"]] = summary
        self.save_file_summaries(summaries)
        return summary

    def invalidate_file_summary(self, path: str | Path, reason: str) -> None:
        relative = self._relative_project_path(path)
        summaries = self.load_file_summaries()
        summary = summaries.get(relative)
        if summary is None:
            summary = {"path": relative, "content_hash": "", "language": "", "imports": [], "symbols": []}
        summary["stale"] = True
        summary["stale_reason"] = reason
        summaries[relative] = summary
        self.save_file_summaries(summaries)

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
        scratchpad = self.load_scratchpad()
        if tool == "read_file" and isinstance(result.get("path"), str):
            _append_unique(scratchpad, "read_files", result["path"])
            self.update_file_summary(result["path"])
        if tool == "write_file" and isinstance(result.get("path"), str):
            _append_unique(scratchpad, "modified_files", result["path"])
            self.invalidate_file_summary(result["path"], "written")
        if tool == "apply_patch":
            for path in result.get("changed_files") or []:
                if isinstance(path, str):
                    _append_unique(scratchpad, "modified_files", path)
                    self.invalidate_file_summary(path, "patched")
        if tool == "run_shell":
            _append_limited(
                scratchpad,
                "last_verified_commands",
                {
                    "command": result.get("command") or arguments.get("command"),
                    "ok": result.get("ok"),
                    "exit_code": result.get("exit_code"),
                    "timed_out": result.get("timed_out"),
                },
                limit=20,
            )
        if result.get("ok") is False:
            _append_limited(
                scratchpad,
                "known_issues",
                {
                    "source": f"tool:{tool}",
                    "error": result.get("error") or result.get("stderr") or "工具执行失败",
                },
                limit=30,
            )
        self.save_scratchpad(scratchpad)
        self.append_tool_index(
            {
                "tool": tool,
                "summary": summarize_tool_result(tool, arguments, result),
                "ok": result.get("ok"),
                "metadata": result.get("metadata") or {},
            }
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


def _normalize_scratchpad(scratchpad: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(DEFAULT_SCRATCHPAD)
    normalized.update(scratchpad)
    for key in ("user_preferences", "confirmed_decisions", "modified_files", "read_files", "known_issues", "active_todos", "last_verified_commands"):
        if not isinstance(normalized.get(key), list):
            normalized[key] = []
    if not isinstance(normalized.get("project_goal"), str):
        normalized["project_goal"] = ""
    return normalized


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


def _append_unique(scratchpad: dict[str, Any], key: str, value: str, *, limit: int = 100) -> None:
    items = scratchpad.setdefault(key, [])
    if not isinstance(items, list):
        items = []
        scratchpad[key] = items
    if value in items:
        items.remove(value)
    items.append(value)
    del items[:-limit]


def _append_limited(scratchpad: dict[str, Any], key: str, value: Any, *, limit: int) -> None:
    items = scratchpad.setdefault(key, [])
    if not isinstance(items, list):
        items = []
        scratchpad[key] = items
    items.append(value)
    del items[:-limit]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
