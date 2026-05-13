from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from coding_agent.sandbox import WorkspaceSandbox
from coding_agent.shell import ShellRunner


ToolFunc = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: ToolFunc


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: ToolFunc,
    ) -> None:
        self._tools[name] = Tool(name=name, description=description, parameters=parameters, func=func)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool: {name}"}

        try:
            return tool.func(arguments)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def create_default_tools(sandbox: WorkspaceSandbox, shell: ShellRunner) -> ToolRegistry:
    registry = ToolRegistry()

    def list_files(arguments: dict[str, Any]) -> dict[str, Any]:
        root = sandbox.resolve(arguments.get("path", "."))
        if not root.exists():
            raise FileNotFoundError(f"path not found: {sandbox.relative_path(root)}")
        paths = [path for path in root.rglob("*") if path.is_file()] if root.is_dir() else [root]
        return {"ok": True, "files": sorted(sandbox.relative_path(path) for path in paths)}

    def read_file(arguments: dict[str, Any]) -> dict[str, Any]:
        path = sandbox.resolve(arguments["path"])
        return {
            "ok": True,
            "path": sandbox.relative_path(path),
            "content": path.read_text(encoding="utf-8"),
        }

    def write_file(arguments: dict[str, Any]) -> dict[str, Any]:
        path = sandbox.resolve(arguments["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments["content"], encoding="utf-8")
        return {"ok": True, "path": sandbox.relative_path(path)}

    def search_text(arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments["query"]
        root = sandbox.resolve(arguments.get("path", "."))
        paths = [path for path in root.rglob("*") if path.is_file()] if root.is_dir() else [root]
        matches: list[dict[str, Any]] = []

        for path in sorted(paths, key=sandbox.relative_path):
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue

            for line_number, line in enumerate(lines, start=1):
                if query in line:
                    matches.append(
                        {
                            "path": sandbox.relative_path(path),
                            "line": line_number,
                            "text": line,
                        }
                    )
                    if len(matches) >= 100:
                        return {"ok": True, "matches": matches}

        return {"ok": True, "matches": matches}

    def apply_patch(arguments: dict[str, Any]) -> dict[str, Any]:
        return {"ok": False, "error": "apply_patch is reserved for later and currently unsupported"}

    def run_shell(arguments: dict[str, Any]) -> dict[str, Any]:
        result = shell.run(arguments["command"])
        return {
            "ok": result.allowed and result.exit_code == 0 and not result.timed_out,
            "command": result.command,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
        }

    registry.register(
        "list_files",
        "List files under a workspace path.",
        _parameters(
            properties={"path": {"type": "string", "default": "."}},
        ),
        list_files,
    )
    registry.register(
        "read_file",
        "Read a UTF-8 file from the workspace.",
        _parameters(
            properties={"path": {"type": "string"}},
            required=["path"],
        ),
        read_file,
    )
    registry.register(
        "write_file",
        "Write UTF-8 content to a workspace file.",
        _parameters(
            properties={"path": {"type": "string"}, "content": {"type": "string"}},
            required=["path", "content"],
        ),
        write_file,
    )
    registry.register(
        "search_text",
        "Search UTF-8 files in the workspace for text.",
        _parameters(
            properties={"query": {"type": "string"}, "path": {"type": "string", "default": "."}},
            required=["query"],
        ),
        search_text,
    )
    registry.register(
        "apply_patch",
        "Apply a patch to workspace files.",
        _parameters(
            properties={"patch": {"type": "string"}},
            required=["patch"],
        ),
        apply_patch,
    )
    registry.register(
        "run_shell",
        "Run a shell command through the command policy.",
        _parameters(
            properties={"command": {"type": "string"}},
            required=["command"],
        ),
        run_shell,
    )

    return registry


def _parameters(
    *,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema
