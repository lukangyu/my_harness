from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

from coding_agent.execution.sandbox import WorkspaceSandbox
from coding_agent.execution.shell import ShellRunner


ToolFunc = Callable[[dict[str, Any]], dict[str, Any]]
DEFAULT_IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".coding-agent",
}
DEFAULT_LIST_MAX_ENTRIES = 200
DEFAULT_READ_MAX_CHARS = 50_000
DEFAULT_SEARCH_MAX_MATCHES = 100
DEFAULT_SESSION_SEARCH_MAX_MATCHES = 50
SESSION_SEARCH_SOURCE_NAMES = {"memory", "sessions", "runs", "dialog", "tool_result"}


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


def create_default_tools(
    sandbox: WorkspaceSandbox,
    shell: ShellRunner,
) -> ToolRegistry:
    registry = ToolRegistry()

    def list_files(arguments: dict[str, Any]) -> dict[str, Any]:
        root = sandbox.resolve(arguments.get("path", "."))
        if not root.exists():
            raise FileNotFoundError(f"path not found: {sandbox.relative_path(root)}")
        max_entries = _positive_int(arguments.get("max_entries"), DEFAULT_LIST_MAX_ENTRIES)
        max_depth = _optional_positive_int(arguments.get("max_depth"))
        paths = _iter_files(root, root, max_depth=max_depth) if root.is_dir() else [root]
        files: list[str] = []
        total_seen = 0
        truncated = False
        for path in sorted(paths, key=sandbox.relative_path):
            total_seen += 1
            if len(files) >= max_entries:
                truncated = True
                continue
            files.append(sandbox.relative_path(path))
        return {
            "ok": True,
            "files": files,
            "metadata": {
                "root": sandbox.relative_path(root),
                "returned_files": len(files),
                "total_seen": total_seen,
                "truncated": truncated,
            },
        }

    def read_file(arguments: dict[str, Any]) -> dict[str, Any]:
        path = sandbox.resolve(arguments["path"])
        start_line = _positive_int(arguments.get("start_line"), 1)
        end_line = _optional_positive_int(arguments.get("end_line"))
        if end_line is not None and end_line < start_line:
            raise ValueError("end_line must be greater than or equal to start_line")
        max_chars = _positive_int(arguments.get("max_chars"), DEFAULT_READ_MAX_CHARS)
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        start_index = min(start_line - 1, len(lines))
        requested_end_index = len(lines) if end_line is None else min(end_line, len(lines))
        content = "".join(lines[start_index:requested_end_index])
        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars]
            truncated = True
        returned_newlines = content.count("\n")
        ends_mid_line = bool(content) and not content.endswith("\n")
        end_index = min(start_index + returned_newlines + (1 if ends_mid_line else 0), len(lines))
        next_start_line = end_index if truncated and ends_mid_line else end_index + 1 if truncated else None
        if next_start_line is not None:
            next_start_line = min(max(next_start_line, start_line), len(lines))
        notice = (
            f"Output truncated at {max_chars} chars. Continue with start_line={next_start_line}."
            if truncated and next_start_line is not None
            else None
        )
        return {
            "ok": True,
            "path": sandbox.relative_path(path),
            "content": content,
            "metadata": {
                "start_line": start_line,
                "end_line": end_index,
                "total_lines": len(lines),
                "returned_chars": len(content),
                "truncated": truncated,
                "next_start_line": next_start_line,
                "notice": notice,
            },
        }

    def write_file(arguments: dict[str, Any]) -> dict[str, Any]:
        path = sandbox.resolve(arguments["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(arguments["content"], encoding="utf-8")
        return {
            "ok": True,
            "path": sandbox.relative_path(path),
            "metadata": {"written_chars": len(arguments["content"])},
        }

    def search_text(arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments["query"]
        case_sensitive = bool(arguments.get("case_sensitive", True))
        use_regex = bool(arguments.get("regex", False))
        glob = arguments.get("glob")
        max_matches = _positive_int(arguments.get("max_matches"), DEFAULT_SEARCH_MAX_MATCHES)
        root = sandbox.resolve(arguments.get("path", "."))
        if not root.exists():
            raise FileNotFoundError(f"path not found: {sandbox.relative_path(root)}")
        rg_result = _search_text_with_rg(
            sandbox,
            root,
            query,
            case_sensitive=case_sensitive,
            use_regex=use_regex,
            glob=glob,
            max_matches=max_matches,
        )
        if rg_result is not None:
            return rg_result
        return _search_text_with_python(
            sandbox,
            root,
            query,
            case_sensitive=case_sensitive,
            use_regex=use_regex,
            glob=glob,
            max_matches=max_matches,
        )

    def apply_patch(arguments: dict[str, Any]) -> dict[str, Any]:
        result = _apply_patch_text(sandbox, arguments["patch"])
        return {"ok": True, **result}

    def run_shell(arguments: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        result = shell.run(arguments["command"])
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "ok": result.allowed and result.exit_code == 0 and not result.timed_out,
            "command": result.command,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "metadata": {"elapsed_ms": elapsed_ms},
        }

    registry.register(
        "list_files",
        "List files under a workspace path.",
        _parameters(
            properties={
                "path": {"type": "string", "default": "."},
                "max_entries": {"type": "integer", "default": DEFAULT_LIST_MAX_ENTRIES},
                "max_depth": {"type": "integer"},
            },
        ),
        list_files,
    )
    registry.register(
        "read_file",
        "Read a UTF-8 file from the workspace.",
        _parameters(
            properties={
                "path": {"type": "string"},
                "start_line": {"type": "integer", "default": 1},
                "end_line": {"type": "integer"},
                "max_chars": {"type": "integer", "default": DEFAULT_READ_MAX_CHARS},
            },
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
            properties={
                "query": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "case_sensitive": {"type": "boolean", "default": True},
                "regex": {"type": "boolean", "default": False},
                "glob": {"type": "string"},
                "max_matches": {"type": "integer", "default": DEFAULT_SEARCH_MAX_MATCHES},
            },
            required=["query"],
        ),
        search_text,
    )
    registry.register(
        "apply_patch",
        "Apply an OpenAI-style patch to workspace files.",
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


def register_session_search_tool(
    registry: ToolRegistry,
    sandbox: WorkspaceSandbox,
    roots: list[Path],
    current_session_root: Path | None = None,
) -> None:
    candidate_roots = [sandbox.resolve(root) for root in roots]
    current_session = sandbox.resolve(current_session_root) if current_session_root is not None else None

    def session_search(arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments["query"]
        case_sensitive = bool(arguments.get("case_sensitive", False))
        use_regex = bool(arguments.get("regex", False))
        glob = arguments.get("glob")
        max_matches = _positive_int(arguments.get("max_matches"), DEFAULT_SESSION_SEARCH_MAX_MATCHES)
        sources = _session_search_sources(arguments.get("sources"))
        scope = arguments.get("scope", "current_conversation")
        if scope not in {"current_session", "current_conversation"}:
            raise ValueError("scope must be current_session or current_conversation")
        root_pool = [current_session] if scope == "current_session" and current_session is not None else candidate_roots
        selected_roots = [
            root
            for root in root_pool
            if root is not None and root.exists()
        ]
        searched_roots = [sandbox.relative_path(root) for root in selected_roots]

        rg_result = _session_search_with_rg(
            sandbox,
            selected_roots,
            query,
            case_sensitive=case_sensitive,
            use_regex=use_regex,
            glob=glob,
            max_matches=max_matches,
            sources=sources,
        )
        if rg_result is not None:
            rg_result["metadata"]["searched_roots"] = searched_roots
            return rg_result

        result = _session_search_with_python(
            sandbox,
            selected_roots,
            query,
            case_sensitive=case_sensitive,
            use_regex=use_regex,
            glob=glob,
            max_matches=max_matches,
            sources=sources,
        )
        result["metadata"]["searched_roots"] = searched_roots
        return result

    registry.register(
        "session_search",
        "Search archived agent memory, sessions, and run artifacts.",
        _parameters(
            properties={
                "query": {"type": "string"},
                "case_sensitive": {"type": "boolean", "default": False},
                "regex": {"type": "boolean", "default": False},
                "glob": {"type": "string"},
                "max_matches": {"type": "integer", "default": DEFAULT_SESSION_SEARCH_MAX_MATCHES},
                "scope": {
                    "type": "string",
                    "enum": ["current_session", "current_conversation"],
                    "default": "current_conversation",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(SESSION_SEARCH_SOURCE_NAMES)},
                },
            },
            required=["query"],
        ),
        session_search,
    )


def _session_search_with_python(
    sandbox: WorkspaceSandbox,
    roots: list[Path],
    query: str,
    *,
    case_sensitive: bool,
    use_regex: bool,
    glob: Any,
    max_matches: int,
    sources: set[str] | None,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    matcher = _compile_matcher(query, case_sensitive=case_sensitive, use_regex=use_regex)

    for root in sorted(roots, key=sandbox.relative_path):
        paths = _iter_memory_files(root) if root.is_dir() else [root]
        for path in sorted(paths, key=sandbox.relative_path):
            relative = sandbox.relative_path(path)
            if isinstance(glob, str) and glob and not fnmatch(Path(relative).name, glob) and not fnmatch(relative, glob):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            for line_number, line in enumerate(lines, start=1):
                if matcher(line):
                    source = _session_source_for_path(relative)
                    if sources is not None and source not in sources:
                        continue
                    matches.append(
                        {
                            "source": source,
                            "path": relative,
                            "line": line_number,
                            "text": line,
                        }
                    )
                    if len(matches) >= max_matches:
                        return {
                            "ok": True,
                            "matches": matches,
                            "metadata": {
                                "engine": "python",
                                "returned_matches": len(matches),
                                "truncated": True,
                                "regex": use_regex,
                                "case_sensitive": case_sensitive,
                            },
                        }

    return {
        "ok": True,
        "matches": matches,
        "metadata": {
            "engine": "python",
            "returned_matches": len(matches),
            "truncated": False,
            "regex": use_regex,
            "case_sensitive": case_sensitive,
        },
    }


def _session_search_with_rg(
    sandbox: WorkspaceSandbox,
    roots: list[Path],
    query: str,
    *,
    case_sensitive: bool,
    use_regex: bool,
    glob: Any,
    max_matches: int,
    sources: set[str] | None,
) -> dict[str, Any] | None:
    rg = shutil.which("rg")
    if rg is None:
        return None

    matches: list[dict[str, Any]] = []
    for root in sorted(roots, key=sandbox.relative_path):
        command = [rg, "--line-number", "--no-heading", "--color=never"]
        if not case_sensitive:
            command.append("--ignore-case")
        if not use_regex:
            command.append("--fixed-strings")
        if isinstance(glob, str) and glob:
            command.extend(["--glob", glob])
        command.extend(["--", query, "."])

        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        except OSError:
            return None

        if result.returncode not in (0, 1):
            return None

        for line in result.stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3:
                continue
            match_path, line_number_text, text = parts
            try:
                line_number = int(line_number_text)
            except ValueError:
                continue
            absolute_match_path = (root / match_path).resolve()
            relative = sandbox.relative_path(absolute_match_path)
            source = _session_source_for_path(relative)
            if sources is not None and source not in sources:
                continue
            matches.append(
                {
                    "source": source,
                    "path": relative,
                    "line": line_number,
                    "text": text,
                }
            )

    matches.sort(key=lambda match: (match["path"], match["line"], match["text"]))
    truncated = len(matches) > max_matches
    returned_matches = matches[:max_matches]

    return {
        "ok": True,
        "matches": returned_matches,
        "metadata": {
            "engine": "rg",
            "returned_matches": len(returned_matches),
            "truncated": truncated,
            "regex": use_regex,
            "case_sensitive": case_sensitive,
        },
    }


def _search_text_with_python(
    sandbox: WorkspaceSandbox,
    root: Path,
    query: str,
    *,
    case_sensitive: bool,
    use_regex: bool,
    glob: Any,
    max_matches: int,
) -> dict[str, Any]:
    paths = _iter_files(root, root, max_depth=None) if root.is_dir() else [root]
    matches: list[dict[str, Any]] = []
    matcher = _compile_matcher(query, case_sensitive=case_sensitive, use_regex=use_regex)

    for path in sorted(paths, key=sandbox.relative_path):
        relative = sandbox.relative_path(path)
        if isinstance(glob, str) and glob and not fnmatch(Path(relative).name, glob) and not fnmatch(relative, glob):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        for line_number, line in enumerate(lines, start=1):
            if matcher(line):
                matches.append(
                    {
                        "path": relative,
                        "line": line_number,
                        "text": line,
                    }
                )
                if len(matches) >= max_matches:
                    return {
                        "ok": True,
                        "matches": matches,
                        "metadata": {
                            "engine": "python",
                            "returned_matches": len(matches),
                            "truncated": True,
                            "regex": use_regex,
                            "case_sensitive": case_sensitive,
                        },
                    }

    return {
        "ok": True,
        "matches": matches,
        "metadata": {
            "engine": "python",
            "returned_matches": len(matches),
            "truncated": False,
            "regex": use_regex,
            "case_sensitive": case_sensitive,
        },
    }


def _search_text_with_rg(
    sandbox: WorkspaceSandbox,
    root: Path,
    query: str,
    *,
    case_sensitive: bool,
    use_regex: bool,
    glob: Any,
    max_matches: int,
) -> dict[str, Any] | None:
    rg = shutil.which("rg")
    if rg is None:
        return None

    search_root = "." if root.is_dir() else root.name
    cwd = root if root.is_dir() else root.parent
    command = [rg, "--line-number", "--no-heading", "--color=never"]
    if not case_sensitive:
        command.append("--ignore-case")
    if not use_regex:
        command.append("--fixed-strings")
    if isinstance(glob, str) and glob:
        command.extend(["--glob", glob])
    command.extend(["--", query, search_root])

    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError:
        return None

    if result.returncode not in (0, 1):
        return None

    matches: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        match_path, line_number_text, text = parts
        try:
            line_number = int(line_number_text)
        except ValueError:
            continue
        absolute_match_path = (cwd / match_path).resolve()
        matches.append(
            {
                "path": sandbox.relative_path(absolute_match_path),
                "line": line_number,
                "text": text,
            }
        )
    matches.sort(key=lambda match: (match["path"], match["line"], match["text"]))
    truncated = len(matches) > max_matches
    returned_matches = matches[:max_matches]

    return {
        "ok": True,
        "matches": returned_matches,
        "metadata": {
            "engine": "rg",
            "returned_matches": len(returned_matches),
            "truncated": truncated,
            "regex": use_regex,
            "case_sensitive": case_sensitive,
        },
    }


def _apply_patch_text(sandbox: WorkspaceSandbox, patch: str) -> dict[str, Any]:
    lines = patch.splitlines()
    if not lines or lines[0] != "*** Begin Patch":
        raise ValueError("patch must start with *** Begin Patch")
    if lines[-1] != "*** End Patch":
        raise ValueError("patch must end with *** End Patch")

    changed_files: list[str] = []
    operations = 0
    index = 1
    while index < len(lines) - 1:
        line = lines[index]
        if line.startswith("*** Add File: "):
            index = _apply_add_file(sandbox, lines, index, changed_files)
            operations += 1
            continue
        if line.startswith("*** Delete File: "):
            path = sandbox.resolve(line.removeprefix("*** Delete File: "))
            path.unlink()
            changed_files.append(sandbox.relative_path(path))
            index += 1
            operations += 1
            continue
        if line.startswith("*** Update File: "):
            index = _apply_update_file(sandbox, lines, index, changed_files)
            operations += 1
            continue
        raise ValueError(f"unsupported patch line: {line}")

    return {"changed_files": changed_files, "metadata": {"operations": operations}}


def _apply_add_file(sandbox: WorkspaceSandbox, lines: list[str], index: int, changed_files: list[str]) -> int:
    path = sandbox.resolve(lines[index].removeprefix("*** Add File: "))
    if path.exists():
        raise FileExistsError(f"file already exists: {sandbox.relative_path(path)}")
    index += 1
    content_lines: list[str] = []
    while index < len(lines) and not _is_patch_boundary(lines[index]):
        line = lines[index]
        if not line.startswith("+"):
            raise ValueError("add file lines must start with +")
        content_lines.append(line[1:])
        index += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content_lines) + ("\n" if content_lines else ""), encoding="utf-8")
    changed_files.append(sandbox.relative_path(path))
    return index


def _apply_update_file(sandbox: WorkspaceSandbox, lines: list[str], index: int, changed_files: list[str]) -> int:
    path = sandbox.resolve(lines[index].removeprefix("*** Update File: "))
    if not path.exists():
        raise FileNotFoundError(f"path not found: {sandbox.relative_path(path)}")
    target_path = path
    index += 1
    if index < len(lines) and lines[index].startswith("*** Move to: "):
        target_path = sandbox.resolve(lines[index].removeprefix("*** Move to: "))
        index += 1

    content = path.read_text(encoding="utf-8")
    had_trailing_newline = content.endswith("\n")
    current_lines = content.splitlines()
    cursor = 0
    while index < len(lines) and not _is_patch_boundary(lines[index]):
        if lines[index].startswith("@@"):
            index += 1
        hunk_lines: list[str] = []
        while index < len(lines) and not lines[index].startswith("@@") and not _is_patch_boundary(lines[index]):
            if lines[index] == "*** End of File":
                index += 1
                continue
            if not lines[index] or lines[index][0] not in " +-":
                raise ValueError(f"unsupported hunk line: {lines[index]}")
            hunk_lines.append(lines[index])
            index += 1
        if not hunk_lines:
            continue
        old_block = [line[1:] for line in hunk_lines if line.startswith((" ", "-"))]
        new_block = [line[1:] for line in hunk_lines if line.startswith((" ", "+"))]
        match_index = _find_block(current_lines, old_block, cursor)
        if match_index is None:
            raise ValueError(f"patch context not found in {sandbox.relative_path(path)}")
        current_lines[match_index : match_index + len(old_block)] = new_block
        cursor = match_index + len(new_block)

    output = "\n".join(current_lines)
    if had_trailing_newline:
        output += "\n"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(output, encoding="utf-8")
    if target_path != path:
        path.unlink()
    changed_files.append(sandbox.relative_path(target_path))
    return index


def _find_block(lines: list[str], block: list[str], start: int) -> int | None:
    if not block:
        return start
    for index in range(start, len(lines) - len(block) + 1):
        if lines[index : index + len(block)] == block:
            return index
    return None


def _is_patch_boundary(line: str) -> bool:
    return (
        line == "*** End Patch"
        or line.startswith("*** Add File: ")
        or line.startswith("*** Delete File: ")
        or line.startswith("*** Update File: ")
    )


def _summarize_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    summarized: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            summarized[key] = {"length": len(value)} if key in {"content", "patch"} else value
        else:
            summarized[key] = value
    return summarized


def _summarize_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("path", "files", "matches", "changed_files", "metadata", "exit_code", "timed_out", "error"):
        if key not in result:
            continue
        value = result[key]
        if isinstance(value, list):
            summary[key] = {"count": len(value), "sample": value[:5]}
        elif isinstance(value, str) and len(value) > 500:
            summary[key] = {"length": len(value), "preview": value[:500]}
        else:
            summary[key] = value
    return summary


def _parameters(
    *,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


def _iter_files(root: Path, base: Path, *, max_depth: int | None) -> list[Path]:
    files: list[Path] = []
    for child in root.iterdir():
        relative = child.relative_to(base)
        if _is_ignored(relative):
            continue
        if child.is_file():
            files.append(child)
            continue
        if child.is_dir():
            if max_depth is not None and len(relative.parts) >= max_depth:
                continue
            files.extend(_iter_files(child, base, max_depth=max_depth))
    return files


def _iter_memory_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for child in root.iterdir():
        if child.name in {"__pycache__", "audit"} or child.name.startswith(".pytest"):
            continue
        if child.is_file():
            files.append(child)
            continue
        if child.is_dir():
            files.extend(_iter_memory_files(child))
    return files


def _session_search_sources(value: Any) -> set[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("sources must be a list of source names")
    sources = set(value)
    unknown = sources - SESSION_SEARCH_SOURCE_NAMES
    if unknown:
        raise ValueError(f"unknown session_search sources: {', '.join(sorted(unknown))}")
    return sources


def _session_source_for_path(relative: str) -> str:
    parts = Path(relative).parts
    if "agent_context" in parts and "dialog" in parts:
        return "dialog"
    if "agent_context" in parts and "tool_result" in parts:
        return "tool_result"
    if "dialog" in parts:
        return "dialog"
    if "tool_result" in parts:
        return "tool_result"
    if len(parts) >= 2 and parts[0] == ".coding-agent":
        if parts[1] == "memory":
            return "memory"
        if parts[1] == "sessions":
            return "sessions"
        if parts[1] == "runs":
            return "runs"
    return "memory"


def _is_ignored(relative: Path) -> bool:
    return any(part in DEFAULT_IGNORED_DIRS or part.startswith(".pytest") for part in relative.parts)


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("expected a positive integer")
    return value


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    return _positive_int(value, 1)


def _compile_matcher(
    query: str,
    *,
    case_sensitive: bool,
    use_regex: bool,
) -> Callable[[str], bool]:
    if use_regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(query, flags)
        return lambda line: pattern.search(line) is not None
    if case_sensitive:
        return lambda line: query in line
    lowered_query = query.lower()
    return lambda line: lowered_query in line.lower()
