from __future__ import annotations

from typing import Any, Callable, Protocol


class MemoryFacade(Protocol):
    def load_scratchpad(self) -> dict[str, Any]:
        ...

    def save_scratchpad(self, scratchpad: dict[str, Any]) -> None:
        ...

    def update_file_summary(self, path: str) -> dict[str, Any] | None:
        ...

    def invalidate_file_summary(self, path: str, reason: str) -> None:
        ...

    def append_tool_index(self, record: dict[str, Any]) -> None:
        ...


class ToolMemoryProjector:
    def __init__(
        self,
        memory_store: MemoryFacade,
        summarize_tool_result: Callable[[str, dict[str, Any], dict[str, Any]], str],
    ) -> None:
        self.memory_store = memory_store
        self.summarize_tool_result = summarize_tool_result

    def record_tool_result(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        scratchpad = self.memory_store.load_scratchpad()
        if tool == "read_file" and isinstance(result.get("path"), str):
            _append_unique(scratchpad, "read_files", result["path"])
            self.memory_store.update_file_summary(result["path"])
        if tool == "write_file" and isinstance(result.get("path"), str):
            _append_unique(scratchpad, "modified_files", result["path"])
            self.memory_store.invalidate_file_summary(result["path"], "written")
        if tool == "apply_patch":
            for path in result.get("changed_files") or []:
                if isinstance(path, str):
                    _append_unique(scratchpad, "modified_files", path)
                    self.memory_store.invalidate_file_summary(path, "patched")
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
        self.memory_store.save_scratchpad(scratchpad)
        self.memory_store.append_tool_index(
            {
                "tool": tool,
                "summary": self.summarize_tool_result(tool, arguments, result),
                "ok": result.get("ok"),
                "metadata": result.get("metadata") or {},
            }
        )


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
