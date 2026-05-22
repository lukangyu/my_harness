from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import ast
import hashlib
import json
import re


MAX_SYMBOLS = 30
MAX_PREVIEW_LINES = 8


@dataclass(frozen=True)
class SymbolSummary:
    type: str
    name: str
    line: int
    preview: list[str] = field(default_factory=list)
    docstring: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type,
            "name": self.name,
            "line": self.line,
            "preview": self.preview,
        }
        if self.docstring:
            data["docstring"] = self.docstring
        return data


@dataclass(frozen=True)
class FileSummary:
    path: str
    content_hash: str
    language: str
    imports: list[str]
    symbols: list[SymbolSummary]
    stale: bool = False
    stale_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "path": self.path,
            "content_hash": self.content_hash,
            "language": self.language,
            "imports": self.imports,
            "symbols": [symbol.to_dict() for symbol in self.symbols],
            "stale": self.stale,
        }
        if self.stale_reason:
            data["stale_reason"] = self.stale_reason
        return data


def summarize_file(path: Path, workspace_root: Path) -> FileSummary:
    relative = path.resolve().relative_to(workspace_root.resolve()).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    language = detect_language(path)
    if language == "python":
        imports, symbols = _summarize_python(text, lines)
    elif language in {"javascript", "typescript"}:
        imports, symbols = _summarize_javascript(lines)
    elif language == "go":
        imports, symbols = _summarize_regex(lines, import_pattern=r"^\s*import\b.*", symbol_pattern=r"^\s*func\s+([A-Za-z_][\w]*)\s*\(")
    elif language == "rust":
        imports, symbols = _summarize_regex(lines, import_pattern=r"^\s*use\s+.*", symbol_pattern=r"^\s*(?:pub\s+)?(?:struct|enum|fn)\s+([A-Za-z_][\w]*)")
    elif language == "markdown":
        imports, symbols = [], _summarize_markdown(lines)
    else:
        imports, symbols = _summarize_regex(lines, import_pattern=r"^\s*(?:import|from|use|require)\b.*", symbol_pattern=r"^\s*(?:class|function|def)\s+([A-Za-z_][\w]*)")
    return FileSummary(
        path=relative,
        content_hash=file_hash(path),
        language=language,
        imports=imports[:50],
        symbols=symbols[:MAX_SYMBOLS],
    )


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_language(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in {".js", ".jsx", ".mjs", ".cjs"}:
        return "javascript"
    if suffix in {".ts", ".tsx"}:
        return "typescript"
    if suffix == ".go":
        return "go"
    if suffix == ".rs":
        return "rust"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    return suffix.removeprefix(".") or "text"


def is_summary_valid(summary: dict[str, Any], path: Path) -> bool:
    if summary.get("stale") is True or not path.is_file():
        return False
    content_hash = summary.get("content_hash")
    return isinstance(content_hash, str) and content_hash == file_hash(path)


class FileSummaryStore:
    def __init__(self, project_root: Path, memory_dir: Path) -> None:
        self.project_root = project_root
        self.memory_dir = memory_dir
        self.path = memory_dir / "file_summaries.json"

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            path: value
            for path, value in data.items()
            if isinstance(path, str) and isinstance(value, dict)
        }

    def save(self, summaries: dict[str, dict[str, Any]]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(summaries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def update(self, path: str | Path) -> dict[str, Any] | None:
        absolute = self._resolve_project_path(path)
        if not absolute.is_file():
            return None
        summary = summarize_file(absolute, self.project_root.resolve()).to_dict()
        summaries = self.load()
        summaries[summary["path"]] = summary
        self.save(summaries)
        return summary

    def invalidate(self, path: str | Path, reason: str) -> None:
        relative = self._relative_project_path(path)
        summaries = self.load()
        summary = summaries.get(relative)
        if summary is None:
            summary = {
                "path": relative,
                "content_hash": "",
                "language": "",
                "imports": [],
                "symbols": [],
            }
        summary["stale"] = True
        summary["stale_reason"] = reason
        summaries[relative] = summary
        self.save(summaries)

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


def _summarize_python(text: str, lines: list[str]) -> tuple[list[str], list[SymbolSummary]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _summarize_regex(lines, import_pattern=r"^\s*(?:import|from)\s+.*", symbol_pattern=r"^\s*(?:class|def|async\s+def)\s+([A-Za-z_][\w]*)")

    imports: list[str] = []
    symbols: list[SymbolSummary] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(_line_at(lines, node.lineno))
            continue
        if isinstance(node, ast.ClassDef):
            symbols.append(_python_symbol("class", node.name, node.lineno, lines, ast.get_docstring(node)))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(
                        _python_symbol(
                            "method" if isinstance(child, ast.FunctionDef) else "async_method",
                            f"{node.name}.{child.name}",
                            child.lineno,
                            lines,
                            ast.get_docstring(child),
                        )
                    )
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                _python_symbol(
                    "function" if isinstance(node, ast.FunctionDef) else "async_function",
                    node.name,
                    node.lineno,
                    lines,
                    ast.get_docstring(node),
                )
            )
    return imports, symbols


def _python_symbol(
    symbol_type: str,
    name: str,
    line: int,
    lines: list[str],
    docstring: str | None,
) -> SymbolSummary:
    return SymbolSummary(
        type=symbol_type,
        name=name,
        line=line,
        preview=_preview(lines, line),
        docstring=_clip_docstring(docstring),
    )


def _summarize_javascript(lines: list[str]) -> tuple[list[str], list[SymbolSummary]]:
    imports: list[str] = []
    symbols: list[SymbolSummary] = []
    patterns = [
        re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"),
        re.compile(r"^\s*(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(?[^=]*\)?\s*=>"),
    ]
    for index, line in enumerate(lines, start=1):
        if re.match(r"^\s*import\b.*", line) or re.match(r"^\s*(?:const|let|var)\s+.*=\s*require\(", line):
            imports.append(line.strip())
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                symbols.append(SymbolSummary(type="symbol", name=match.group(1), line=index, preview=_preview(lines, index)))
                break
    return imports, symbols


def _summarize_regex(lines: list[str], *, import_pattern: str, symbol_pattern: str) -> tuple[list[str], list[SymbolSummary]]:
    imports: list[str] = []
    symbols: list[SymbolSummary] = []
    import_re = re.compile(import_pattern)
    symbol_re = re.compile(symbol_pattern)
    for index, line in enumerate(lines, start=1):
        if import_re.match(line):
            imports.append(line.strip())
        match = symbol_re.match(line)
        if match:
            symbols.append(SymbolSummary(type="symbol", name=match.group(1), line=index, preview=_preview(lines, index)))
    return imports, symbols


def _summarize_markdown(lines: list[str]) -> list[SymbolSummary]:
    symbols: list[SymbolSummary] = []
    for index, line in enumerate(lines, start=1):
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            symbols.append(SymbolSummary(type=f"heading{len(match.group(1))}", name=match.group(2).strip(), line=index, preview=[line]))
    return symbols


def _preview(lines: list[str], line: int) -> list[str]:
    start = max(0, line - 1)
    return lines[start : start + MAX_PREVIEW_LINES]


def _line_at(lines: list[str], line: int) -> str:
    if line <= 0 or line > len(lines):
        return ""
    return lines[line - 1].strip()


def _clip_docstring(docstring: str | None, max_chars: int = 500) -> str | None:
    if not docstring:
        return None
    normalized = docstring.strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "... [truncated]"
