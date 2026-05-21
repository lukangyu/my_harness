from __future__ import annotations

import hashlib
import html
import json
import os
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from coding_agent.memory import MemoryStore


DOC_NAMES = (
    "AGENTS.md",
    ".cursorrules",
    "README.md",
    "pyproject.toml",
    "package.json",
    "Makefile",
    "justfile",
    "pytest.ini",
    "tox.ini",
    "Cargo.toml",
    "go.mod",
)

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".coding-agent",
}

PROMPT_VERSION = "context-v1"
STABLE_PREFIX_TEMPLATE = """<coding_agent_prefix version="context-v1">
<identity>
You are coding-agent, a local AI coding assistant.
</identity>

<operating_principles>
- Inspect relevant files before editing.
- Prefer small, focused changes.
- Verify changes with allowed commands when practical.
- Do not claim tests passed unless tool results show they passed.
</operating_principles>

<safety_contract>
- Work only through provided tools.
- File access is limited by the workspace sandbox.
- Shell commands are subject to command policy.
- If a tool is rejected or fails, report the reason and adapt.
</safety_contract>

<tool_contract>
- Use list_files/read_file/search_text to inspect.
- Use write_file only for intentional file writes.
- run_shell may be rejected by policy.
- Use apply_patch for focused workspace file edits when possible.
</tool_contract>

<response_contract>
- Final answers should summarize changes, files touched, verification, and residual risk.
</response_contract>
</coding_agent_prefix>"""


@dataclass(frozen=True)
class WorkspaceContextOptions:
    doc_max_chars: int = 1200
    tree_max_entries: int = 200
    include_project_docs: bool = True
    include_file_tree: bool = True
    include_git_status: bool = True
    include_recent_commits: bool = True
    max_input_tokens: int = 24000
    compact_threshold_ratio: float = 0.8
    protected_recent_turns: int = 4
    protected_tool_results: int = 6
    handoff_max_chars: int = 6000
    scratchpad_max_chars: int = 4000
    file_summaries_max_count: int = 8
    file_summaries_max_chars: int = 8000


class CompactClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class MessagePartitions:
    compactable: list[dict[str, Any]]
    reserved: list[dict[str, Any]]


@dataclass(frozen=True)
class UsageStats:
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None

    @property
    def cache_hit_ratio(self) -> float | None:
        if self.input_tokens and self.cached_tokens is not None:
            return self.cached_tokens / self.input_tokens
        return None

    @classmethod
    def from_response_usage(cls, usage: Any) -> "UsageStats | None":
        if usage is None:
            return None

        prompt_tokens_details = _usage_value(usage, "prompt_tokens_details")
        direct_cached_tokens = _usage_value(usage, "cached_tokens")
        cached_tokens = direct_cached_tokens if isinstance(direct_cached_tokens, int) else None
        if cached_tokens is None and prompt_tokens_details is not None:
            detail_cached_tokens = _usage_value(prompt_tokens_details, "cached_tokens")
            cached_tokens = (
                detail_cached_tokens if isinstance(detail_cached_tokens, int) else None
            )

        return cls(
            input_tokens=_first_usage_value(usage, "prompt_tokens", "input_tokens"),
            output_tokens=_first_usage_value(usage, "completion_tokens", "output_tokens"),
            cached_tokens=cached_tokens,
        )


@dataclass(frozen=True)
class WorkspaceContext:
    cwd: Path
    repo_root: Path | None
    branch: str | None
    default_branch: str | None
    status: str
    recent_commits: list[str]
    project_docs: dict[str, str]
    file_tree: list[str]

    @classmethod
    def build(cls, cwd: Path, options: WorkspaceContextOptions) -> "WorkspaceContext":
        cwd = cwd.resolve()
        repo_root_text = _git(cwd, "rev-parse", "--show-toplevel")
        repo_root = Path(repo_root_text).resolve() if repo_root_text else None
        git_cwd = repo_root or cwd

        branch = _git(git_cwd, "branch", "--show-current") if repo_root else ""
        default_branch = _default_branch(git_cwd) if repo_root else None
        status = (
            _git(git_cwd, "status", "--short")
            if repo_root and options.include_git_status
            else ""
        )
        commits_text = (
            _git(git_cwd, "log", "--oneline", "-5")
            if repo_root and options.include_recent_commits
            else ""
        )

        return cls(
            cwd=cwd,
            repo_root=repo_root,
            branch=branch or None,
            default_branch=default_branch,
            status=status,
            recent_commits=commits_text.splitlines() if commits_text else [],
            project_docs=_project_docs(cwd, repo_root, options)
            if options.include_project_docs
            else {},
            file_tree=_file_tree(repo_root or cwd, options) if options.include_file_tree else [],
        )

    def fingerprint(self) -> str:
        return _hash_json(
            {
                "cwd": str(self.cwd),
                "repo_root": str(self.repo_root) if self.repo_root else None,
                "branch": self.branch,
                "default_branch": self.default_branch,
                "status": self.status,
                "recent_commits": self.recent_commits,
                "project_docs": self.project_docs,
                "file_tree": self.file_tree,
            }
        )


@dataclass(frozen=True)
class StablePrefixState:
    text: str
    system_hash: str
    tool_signature: str
    rules_hash: str
    prompt_cache_key: str
    prompt_version: str


class StablePrefixManager:
    def __init__(self) -> None:
        self._state: StablePrefixState | None = None

    def get_or_build(self, tool_schemas: list[dict[str, Any]]) -> StablePrefixState:
        system_hash = _hash_text(STABLE_PREFIX_TEMPLATE)
        tool_signature = _hash_json(tool_schemas)
        rules_hash = _hash_text("")
        prompt_cache_key = _hash_text(
            PROMPT_VERSION + system_hash + tool_signature + rules_hash
        )
        if self._state and self._state.prompt_cache_key == prompt_cache_key:
            return self._state
        self._state = StablePrefixState(
            text=STABLE_PREFIX_TEMPLATE,
            system_hash=system_hash,
            tool_signature=tool_signature,
            rules_hash=rules_hash,
            prompt_cache_key=prompt_cache_key,
            prompt_version=PROMPT_VERSION,
        )
        return self._state


@dataclass(frozen=True)
class WorkspacePrefixState:
    text: str
    workspace_fingerprint: str
    cwd: str
    repo_root: str | None
    branch: str | None
    default_branch: str | None
    status_hash: str
    recent_commits_hash: str
    project_docs_hash: str
    file_tree_hash: str


@dataclass(frozen=True)
class ContextEnvelope:
    stable_prefix: StablePrefixState
    workspace_prefix: WorkspacePrefixState
    mode: str
    session_summary: str | None
    recent_messages: list[dict[str, Any]]
    current_task: str
    full_context_key: str
    memory_anchor: str = ""
    handoff_memo: str = ""
    file_summaries: str = ""


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


class MessageBudget:
    def __init__(
        self,
        recent_message_tokens: int,
        max_tool_content_chars: int = 4000,
    ) -> None:
        self.recent_message_tokens = recent_message_tokens
        self.max_tool_content_chars = max_tool_content_chars

    def trim_recent_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.recent_message_tokens <= 0:
            return []

        selected_groups: list[list[dict[str, Any]]] = []
        used_tokens = 0
        for group in reversed(self._message_groups(messages)):
            if not group:
                continue
            tokens = sum(self._message_tokens(message) for message in group)
            if used_tokens + tokens > self.recent_message_tokens:
                break
            selected_groups.append(group)
            used_tokens += tokens
        selected_groups.reverse()
        return [
            message
            for group in selected_groups
            for message in group
        ]

    def _message_groups(self, messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        prepared_messages = [self._prepare_message(message) for message in messages]
        groups: list[list[dict[str, Any]]] = []
        index = 0
        while index < len(prepared_messages):
            message = prepared_messages[index]
            if message.get("role") == "tool":
                index += 1
                continue

            tool_call_ids = self._tool_call_ids(message)
            if not tool_call_ids:
                groups.append([message])
                index += 1
                continue

            group = [message]
            found_ids: set[str] = set()
            next_index = index + 1
            while next_index < len(prepared_messages):
                next_message = prepared_messages[next_index]
                if next_message.get("role") != "tool":
                    break
                tool_call_id = next_message.get("tool_call_id")
                if tool_call_id not in tool_call_ids or tool_call_id in found_ids:
                    break
                group.append(next_message)
                found_ids.add(tool_call_id)
                next_index += 1

            if found_ids == tool_call_ids:
                groups.append(group)
                index = next_index
            else:
                index += 1
        return groups

    def _prepare_message(self, message: dict[str, Any]) -> dict[str, Any]:
        prepared = deepcopy(message)
        content = prepared.get("content")
        if (
            prepared.get("role") == "tool"
            and isinstance(content, str)
            and len(content) > self.max_tool_content_chars
        ):
            prepared["content"] = (
                content[: self.max_tool_content_chars]
                + "\n... [truncated by context budget]"
            )
        return prepared

    def _message_tokens(self, message: dict[str, Any]) -> int:
        text = json.dumps(message, sort_keys=True, ensure_ascii=False)
        return estimate_tokens(text)

    def _tool_call_ids(self, message: dict[str, Any]) -> set[str]:
        if message.get("role") != "assistant":
            return set()
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            return set()
        return {
            tool_call["id"]
            for tool_call in tool_calls
            if isinstance(tool_call, dict) and isinstance(tool_call.get("id"), str)
        }


class ContextManager:
    def __init__(
        self,
        cwd: Path,
        options: WorkspaceContextOptions,
        recent_message_tokens: int,
        memory_store: MemoryStore | None = None,
        compact_client: CompactClient | None = None,
    ) -> None:
        self.cwd = cwd
        self.options = options
        self.memory_store = memory_store
        self.compact_client = compact_client
        self.stable_prefix_manager = StablePrefixManager()
        self.workspace_prefix_manager = WorkspacePrefixManager()
        self.message_budget = MessageBudget(recent_message_tokens)

    def build(
        self,
        task: str,
        prior_messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        mode: str = "run",
    ) -> ContextEnvelope:
        stable_prefix = self.stable_prefix_manager.get_or_build(tool_schemas)
        workspace_context = WorkspaceContext.build(self.cwd, self.options)
        workspace_prefix = self.workspace_prefix_manager.get_or_build(workspace_context)
        memory_anchor = self._memory_anchor()
        file_summaries = self._file_summaries()
        handoff_memo = self._handoff_memo()
        compacted_prior_messages, handoff_memo = self._compact_if_needed(
            prior_messages=list(prior_messages),
            stable_prefix=stable_prefix,
            workspace_prefix=workspace_prefix,
            memory_anchor=memory_anchor,
            file_summaries=file_summaries,
            handoff_memo=handoff_memo,
            task=task,
            mode=mode,
        )
        recent_messages = self.message_budget.trim_recent_messages(compacted_prior_messages)
        current_task = str(task)
        full_context_key = _hash_json(
            {
                "stable_prompt_key": stable_prefix.prompt_cache_key,
                "workspace_fingerprint": workspace_prefix.workspace_fingerprint,
                "memory_hash": _hash_text(memory_anchor),
                "file_summaries_hash": _hash_text(file_summaries),
                "handoff_hash": _hash_text(handoff_memo),
                "recent_hash": _hash_json(recent_messages),
                "task_hash": _hash_text(current_task),
                "mode": mode,
            }
        )
        return ContextEnvelope(
            stable_prefix=stable_prefix,
            workspace_prefix=workspace_prefix,
            mode=mode,
            memory_anchor=memory_anchor,
            handoff_memo=handoff_memo,
            file_summaries=file_summaries,
            session_summary=None,
            recent_messages=recent_messages,
            current_task=current_task,
            full_context_key=full_context_key,
        )

    def _memory_anchor(self) -> str:
        if self.memory_store is None:
            return ""
        return self.memory_store.render_memory_anchor(max_chars=self.options.scratchpad_max_chars)

    def _handoff_memo(self) -> str:
        if self.memory_store is None:
            return ""
        handoff = self.memory_store.read_handoff()
        if len(handoff) > self.options.handoff_max_chars:
            return handoff[: self.options.handoff_max_chars] + "\n... [handoff_memo truncated]"
        return handoff

    def _file_summaries(self) -> str:
        if self.memory_store is None:
            return ""
        return self.memory_store.render_file_summaries(
            candidate_paths=self.memory_store.candidate_summary_paths(),
            max_count=self.options.file_summaries_max_count,
            max_chars=self.options.file_summaries_max_chars,
        )

    def _compact_if_needed(
        self,
        *,
        prior_messages: list[dict[str, Any]],
        stable_prefix: StablePrefixState,
        workspace_prefix: WorkspacePrefixState,
        memory_anchor: str,
        file_summaries: str,
        handoff_memo: str,
        task: str,
        mode: str,
    ) -> tuple[list[dict[str, Any]], str]:
        if self.memory_store is None or self.compact_client is None or not prior_messages:
            return prior_messages, handoff_memo
        estimated_tokens = self._estimate_context_tokens(
            stable_prefix=stable_prefix,
            workspace_prefix=workspace_prefix,
            memory_anchor=memory_anchor,
            file_summaries=file_summaries,
            handoff_memo=handoff_memo,
            prior_messages=prior_messages,
            task=task,
            mode=mode,
        )
        threshold = int(self.options.max_input_tokens * self.options.compact_threshold_ratio)
        if estimated_tokens < threshold:
            return prior_messages, handoff_memo

        partitions = partition_messages(
            prior_messages,
            protected_recent_turns=self.options.protected_recent_turns,
            protected_tool_results=self.options.protected_tool_results,
        )
        dialog_path = self.memory_store.archive_dialog_messages(partitions.compactable)
        source_refs = {
            "dialog_path": dialog_path.relative_to(self.memory_store.project_root).as_posix() if dialog_path else None,
            "tool_index_path": self.memory_store.tool_index_path.relative_to(self.memory_store.project_root).as_posix(),
            "tool_result_dir": self.memory_store.tool_result_dir.relative_to(self.memory_store.project_root).as_posix(),
        }
        compact_prompt = build_handoff_prompt(
            old_messages=partitions.compactable,
            previous_handoff=handoff_memo,
            memory_anchor=memory_anchor,
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
        new_handoff = message.get("content") if isinstance(message, dict) else ""
        if not isinstance(new_handoff, str) or not new_handoff.strip():
            return clear_old_tool_results(partitions.reserved), handoff_memo
        new_handoff = new_handoff.strip()
        new_handoff = _prepend_handoff_sources(new_handoff, source_refs)
        if len(new_handoff) > self.options.handoff_max_chars:
            new_handoff = new_handoff[: self.options.handoff_max_chars] + "\n... [handoff_memo truncated]"
        self.memory_store.write_handoff(new_handoff)
        return clear_old_tool_results(partitions.reserved), new_handoff

    def _estimate_context_tokens(
        self,
        *,
        stable_prefix: StablePrefixState,
        workspace_prefix: WorkspacePrefixState,
        memory_anchor: str,
        file_summaries: str,
        handoff_memo: str,
        prior_messages: list[dict[str, Any]],
        task: str,
        mode: str,
    ) -> int:
        messages = [
            {"role": "system", "content": stable_prefix.text},
            {"role": "user", "content": workspace_prefix.text},
        ]
        if memory_anchor:
            messages.append({"role": "user", "content": memory_anchor})
        if file_summaries:
            messages.append({"role": "user", "content": file_summaries})
        if handoff_memo:
            messages.append({"role": "user", "content": handoff_memo})
        messages.extend(prior_messages)
        messages.append({"role": "user", "content": f"<current_task>\nmode: {mode}\ncontent:\n{task}\n</current_task>"})
        return estimate_tokens(json.dumps(messages, ensure_ascii=False, sort_keys=True))


class PromptBuilder:
    def to_messages(
        self,
        envelope: ContextEnvelope,
        mode: str | None = None,
    ) -> list[dict[str, Any]]:
        render_mode = envelope.mode
        if mode is not None and mode != render_mode:
            raise ValueError(
                f"render mode {mode!r} does not match envelope mode {render_mode!r}"
            )
        messages = [
            {"role": "system", "content": envelope.stable_prefix.text},
            {"role": "user", "content": envelope.workspace_prefix.text},
        ]
        if envelope.memory_anchor:
            messages.append({"role": "user", "content": envelope.memory_anchor})
        if envelope.handoff_memo:
            messages.append(
                {
                    "role": "user",
                    "content": f"<handoff_memo>\n{envelope.handoff_memo}\n</handoff_memo>",
                }
            )
        if envelope.file_summaries:
            messages.append({"role": "user", "content": envelope.file_summaries})
        if envelope.session_summary:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"<session_summary>\n"
                        f"{envelope.session_summary}\n"
                        f"</session_summary>"
                    ),
                }
            )
        messages.extend(envelope.recent_messages)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"<current_task>\n"
                    f"mode: {render_mode}\n"
                    f"content:\n{envelope.current_task}\n"
                    f"</current_task>"
                ),
            }
        )
        return messages


def partition_messages(
    messages: list[dict[str, Any]],
    *,
    protected_recent_turns: int,
    protected_tool_results: int,
) -> MessagePartitions:
    if protected_recent_turns <= 0:
        return MessagePartitions(compactable=list(messages), reserved=[])
    protected: list[dict[str, Any]] = []
    protected_tool_count = 0
    user_turns = 0
    cutoff = len(messages)
    for message in reversed(messages):
        cutoff -= 1
        prepared = deepcopy(message)
        if prepared.get("role") == "tool":
            protected_tool_count += 1
            if protected_tool_count > protected_tool_results:
                prepared["content"] = "[旧 tool 输出已清理，关键结论见 handoff_memo 或 tool_index.jsonl]"
            protected.append(prepared)
            continue
        protected.append(prepared)
        if prepared.get("role") == "user":
            user_turns += 1
            if user_turns >= protected_recent_turns:
                break
    protected.reverse()
    return MessagePartitions(compactable=list(messages[:cutoff]), reserved=protected)


def protect_recent_messages(
    messages: list[dict[str, Any]],
    *,
    protected_recent_turns: int,
    protected_tool_results: int,
) -> list[dict[str, Any]]:
    return partition_messages(
        messages,
        protected_recent_turns=protected_recent_turns,
        protected_tool_results=protected_tool_results,
    ).reserved


def clear_old_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleared: list[dict[str, Any]] = []
    for message in messages:
        prepared = deepcopy(message)
        if prepared.get("role") == "tool":
            content = prepared.get("content")
            if isinstance(content, str) and len(content) > 4000:
                prepared["content"] = content[:4000] + "\n... [tool result truncated after compaction]"
        cleared.append(prepared)
    return cleared


def build_handoff_prompt(
    *,
    old_messages: list[dict[str, Any]],
    previous_handoff: str,
    memory_anchor: str,
    source_refs: dict[str, str | None],
) -> str:
    return (
        "你正在进行上下文检查点压缩。请为即将接手任务的下一个 LLM 实例生成一份清晰、精简、可执行的交接摘要。\n"
        "必须包含：\n"
        "1. 当前项目最新进度与核心决策。\n"
        "2. 必须遵守的代码约束与用户偏好。\n"
        "3. 已经读过/修改过/验证过的关键文件和结论。\n"
        "4. 已排除的错误路径，避免重复探索。\n"
        "5. 剩余 TODO 和下一步最小行动。\n\n"
        "输出格式必须是 Markdown，标题使用：当前目标、已完成、关键事实、用户偏好、剩余 TODO、下一步。\n\n"
        "<source_refs>\n"
        f"{json.dumps(source_refs, ensure_ascii=False)}\n"
        "</source_refs>\n\n"
        f"<previous_handoff>\n{previous_handoff or 'none'}\n</previous_handoff>\n\n"
        f"{memory_anchor or '<memory_anchor>none</memory_anchor>'}\n\n"
        "<old_messages_json>\n"
        f"{json.dumps(old_messages, ensure_ascii=False)}\n"
        "</old_messages_json>"
    )


def _prepend_handoff_sources(handoff: str, source_refs: dict[str, str | None]) -> str:
    lines = ["## Source References"]
    for key, value in source_refs.items():
        if value:
            lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append(handoff)
    return "\n".join(lines)


class WorkspacePrefixManager:
    def __init__(self) -> None:
        self._state: WorkspacePrefixState | None = None

    def get_or_build(self, context: WorkspaceContext) -> WorkspacePrefixState:
        fingerprint = context.fingerprint()
        if self._state and self._state.workspace_fingerprint == fingerprint:
            return self._state
        text = render_workspace_context(context)
        self._state = WorkspacePrefixState(
            text=text,
            workspace_fingerprint=fingerprint,
            cwd=str(context.cwd),
            repo_root=str(context.repo_root) if context.repo_root else None,
            branch=context.branch,
            default_branch=context.default_branch,
            status_hash=_hash_text(context.status),
            recent_commits_hash=_hash_json(context.recent_commits),
            project_docs_hash=_hash_json(context.project_docs),
            file_tree_hash=_hash_json(context.file_tree),
        )
        return self._state


def render_workspace_context(context: WorkspaceContext) -> str:
    lines = [
        "<workspace_context>",
        f"cwd: {context.cwd}",
        f"repo_root: {context.repo_root if context.repo_root else ''}",
        f"is_git_repo: {str(context.repo_root is not None).lower()}",
        f"branch: {context.branch or ''}",
        f"default_branch: {context.default_branch or ''}",
        "",
        "git_status:",
    ]
    lines.extend([f"  {line}" for line in context.status.splitlines()] or ["  clean"])
    lines.append("")
    lines.append("recent_commits:")
    lines.extend([f"  {line}" for line in context.recent_commits] or ["  none"])
    lines.append("")
    lines.append("file_tree:")
    lines.extend([f"  {line}" for line in context.file_tree] or ["  none"])
    lines.append("")
    lines.append("project_docs:")
    if context.project_docs:
        for path, content in sorted(context.project_docs.items()):
            lines.append(f'  <doc path="{html.escape(path)}">')
            lines.extend(f"  {html.escape(line)}" for line in content.splitlines())
            lines.append("  </doc>")
    else:
        lines.append("  none")
    lines.append("</workspace_context>")
    return "\n".join(lines)


def _git(cwd: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=5,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _default_branch(cwd: Path) -> str | None:
    origin_head = _git(cwd, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    if origin_head.startswith("origin/"):
        return origin_head.removeprefix("origin/")
    for name in ("main", "master"):
        if _git(cwd, "rev-parse", "--verify", name):
            return name
    return None


def _project_docs(
    cwd: Path, repo_root: Path | None, options: WorkspaceContextOptions
) -> dict[str, str]:
    docs: dict[str, str] = {}
    root = repo_root or cwd
    for base in (repo_root, cwd):
        if base is None:
            continue
        for name in DOC_NAMES:
            path = base / name
            if not path.is_file():
                continue
            try:
                key = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                key = path.name
            if key in docs:
                continue
            try:
                docs[key] = _clip(
                    path.read_text(encoding="utf-8", errors="replace"),
                    options.doc_max_chars,
                )
            except OSError:
                continue
    return docs


def _file_tree(root: Path, options: WorkspaceContextOptions) -> list[str]:
    if options.tree_max_entries <= 0:
        return []

    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        relative_dir = current.relative_to(root)
        if _is_ignored(relative_dir):
            dirnames[:] = []
            continue
        dirnames[:] = sorted(name for name in dirnames if not _is_ignored(relative_dir / name))
        for filename in sorted(filenames):
            path = current / filename
            relative = path.relative_to(root)
            if _is_ignored(relative):
                continue
            files.append(relative.as_posix())
            if len(files) >= options.tree_max_entries:
                return files
    return files


def _is_ignored(relative: Path) -> bool:
    return any(part in IGNORED_DIRS or part.startswith(".pytest") for part in relative.parts)


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _usage_value(usage: Any, key: str) -> Any:
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return value
    return None


def _first_usage_value(usage: Any, *keys: str) -> int | None:
    for key in keys:
        value = _usage_value(usage, key)
        if isinstance(value, int):
            return value
    return None
