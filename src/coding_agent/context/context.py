from __future__ import annotations

import hashlib
import json
import os
import subprocess
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

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

FrameStability = Literal["static", "medium", "dynamic"]


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
    compact_tail_ratio: float = 0.2
    protected_recent_turns: int = 4
    protected_tool_results: int = 6
    handoff_max_chars: int = 6000
    scratchpad_max_chars: int = 4000


class CompactClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        ...


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
            cached_tokens = detail_cached_tokens if isinstance(detail_cached_tokens, int) else None

        return cls(
            input_tokens=_first_usage_value(usage, "prompt_tokens", "input_tokens"),
            output_tokens=_first_usage_value(usage, "completion_tokens", "output_tokens"),
            cached_tokens=cached_tokens,
        )


@dataclass(frozen=True)
class WorkspaceSnapshot:
    cwd: Path
    repo_root: Path | None
    branch: str | None
    default_branch: str | None
    status: str
    recent_commits: list[str]
    project_docs: dict[str, str]
    file_tree: list[str]

    @classmethod
    def build(cls, cwd: Path, options: WorkspaceContextOptions) -> "WorkspaceSnapshot":
        cwd = cwd.resolve()
        repo_root_text = _git(cwd, "rev-parse", "--show-toplevel")
        repo_root = Path(repo_root_text).resolve() if repo_root_text else None
        git_cwd = repo_root or cwd

        branch = _git(git_cwd, "branch", "--show-current") if repo_root else ""
        default_branch = _default_branch(git_cwd) if repo_root else None
        status = _git(git_cwd, "status", "--short") if repo_root and options.include_git_status else ""
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
        return _hash_json(self.to_payload())

    def to_payload(self) -> dict[str, Any]:
        return {
            "cwd": str(self.cwd),
            "repo_root": str(self.repo_root) if self.repo_root else None,
            "branch": self.branch,
            "default_branch": self.default_branch,
            "is_git_repo": self.repo_root is not None,
            "status": self.status,
            "recent_commits": list(self.recent_commits),
            "project_docs": dict(self.project_docs),
            "file_tree": list(self.file_tree),
            "fingerprint": self.fingerprint_without_payload_cycle(),
        }

    def fingerprint_without_payload_cycle(self) -> str:
        value = {
            "cwd": str(self.cwd),
            "repo_root": str(self.repo_root) if self.repo_root else None,
            "branch": self.branch,
            "default_branch": self.default_branch,
            "status": self.status,
            "recent_commits": self.recent_commits,
            "project_docs": self.project_docs,
            "file_tree": self.file_tree,
        }
        return _hash_json(value)

    def with_updates(self, **updates: Any) -> "WorkspaceSnapshot":
        return replace(self, **updates)


@dataclass(frozen=True)
class ContextFrame:
    kind: str
    role: str | None
    payload: dict[str, Any]
    priority: int
    stability: FrameStability
    token_estimate: int


@dataclass(frozen=True)
class MessagePartitions:
    compactable: list[dict[str, Any]]
    reserved: list[dict[str, Any]]


class Context:
    def __init__(
        self,
        *,
        cwd: Path,
        options: WorkspaceContextOptions,
        task: str,
        prior_messages: list[dict[str, Any]] | None = None,
        tool_schemas: list[dict[str, Any]] | None = None,
        mode: str = "run",
        workspace_snapshot: WorkspaceSnapshot | None = None,
        scratchpad: dict[str, Any] | None = None,
        handoff: str = "",
    ) -> None:
        self.cwd = cwd
        self.options = options
        self.task = str(task)
        self.mode = mode
        self.tool_schemas = deepcopy(tool_schemas or [])
        self.workspace_snapshot = workspace_snapshot or WorkspaceSnapshot.build(cwd, options)
        self.scratchpad = deepcopy(scratchpad or {})
        self.handoff = handoff
        self._raw_frames: list[ContextFrame] = []
        self._active_frames: list[ContextFrame] = []
        self._static_frames: list[ContextFrame] = []
        self._build_static_frames()
        for message in prior_messages or []:
            self.add_message(message)

    def add_message(self, message: dict[str, Any]) -> None:
        frame = self._message_frame(message)
        self._raw_frames.append(frame)
        self._active_frames.append(frame)

    def update_workspace_snapshot(self, snapshot: WorkspaceSnapshot) -> None:
        self.workspace_snapshot = snapshot
        self._build_static_frames()

    def estimate_active_tokens(self) -> int:
        payload = {
            "context": [frame.payload for frame in self._static_frames],
            "history": [frame.payload for frame in self._active_frames],
            "task": {"mode": self.mode, "content": self.task},
        }
        return estimate_tokens(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def slice_old_history(
        self,
        *,
        tail_token_budget: int | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        partitions = partition_messages(
            [frame.payload for frame in self._active_frames if frame.kind == "history"],
            protected_recent_turns=self.options.protected_recent_turns,
            protected_tool_results=self.options.protected_tool_results,
            tail_token_budget=tail_token_budget,
        )
        return partitions.compactable, partitions.reserved

    def replace_active_history(
        self,
        summary_payload: dict[str, Any] | None,
        remaining_messages: list[dict[str, Any]],
    ) -> None:
        frames: list[ContextFrame] = []
        if summary_payload is not None:
            frames.append(self._summary_frame(summary_payload))
        frames.extend(self._message_frame(message) for message in remaining_messages)
        self._active_frames = frames

    def add_summary_frame(self, summary_payload: dict[str, Any]) -> None:
        self._active_frames.insert(0, self._summary_frame(summary_payload))

    def frames(self) -> list[ContextFrame]:
        return [
            self._copy_frame(frame)
            for frame in [*self._static_frames, *self._active_frames, self._task_frame()]
        ]

    def raw_messages(self) -> list[dict[str, Any]]:
        return [deepcopy(frame.payload) for frame in self._raw_frames if frame.kind == "history"]

    def history_frames(self) -> list[ContextFrame]:
        return [self._copy_frame(frame) for frame in self._active_frames]

    def context_frames(self) -> list[ContextFrame]:
        return [self._copy_frame(frame) for frame in self._static_frames]

    def task_frame(self) -> ContextFrame:
        return self._copy_frame(self._task_frame())

    def full_context_key(self) -> str:
        return _hash_json(
            {
                "workspace": self.workspace_snapshot.fingerprint(),
                "context": [frame.payload for frame in self._static_frames],
                "history": [frame.payload for frame in self._active_frames],
                "task": {"mode": self.mode, "content": self.task},
                "tools": self.tool_schemas,
            }
        )

    def _build_static_frames(self) -> None:
        frames = [
            self._frame(
                kind="workspace",
                role=None,
                payload=self.workspace_snapshot.to_payload(),
                priority=90,
                stability="medium",
            )
        ]
        if self.scratchpad:
            frames.append(
                self._frame(
                    kind="memory",
                    role=None,
                    payload={"scratchpad": self.scratchpad},
                    priority=80,
                    stability="medium",
                )
            )
        if self.handoff:
            frames.append(
                self._frame(
                    kind="handoff",
                    role=None,
                    payload={"text": _clip(self.handoff, self.options.handoff_max_chars, "handoff_memo")},
                    priority=85,
                    stability="medium",
                )
            )
        self._static_frames = frames

    def _estimate_total_tokens(self, history_messages: list[dict[str, Any]]) -> int:
        payload = {
            "context": [frame.payload for frame in self._static_frames],
            "history": history_messages,
            "task": {"mode": self.mode, "content": self.task},
        }
        return estimate_tokens(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def _message_frame(self, message: dict[str, Any]) -> ContextFrame:
        payload = deepcopy(message)
        return self._frame(
            kind="history",
            role=str(payload.get("role")) if payload.get("role") is not None else None,
            payload=payload,
            priority=50,
            stability="dynamic",
        )

    def _summary_frame(self, summary_payload: dict[str, Any]) -> ContextFrame:
        return self._frame(
            kind="compact_summary",
            role="assistant",
            payload={
                "role": "assistant",
                "content": _format_compact_summary(summary_payload),
            },
            priority=70,
            stability="medium",
        )

    def _task_frame(self) -> ContextFrame:
        return self._frame(
            kind="current_task",
            role="user",
            payload={"mode": self.mode, "content": self.task},
            priority=100,
            stability="dynamic",
        )

    def _frame(
        self,
        *,
        kind: str,
        role: str | None,
        payload: dict[str, Any],
        priority: int,
        stability: FrameStability,
    ) -> ContextFrame:
        copied = deepcopy(payload)
        return ContextFrame(
            kind=kind,
            role=role,
            payload=copied,
            priority=priority,
            stability=stability,
            token_estimate=estimate_tokens(json.dumps(copied, ensure_ascii=False, sort_keys=True)),
        )

    def _copy_frame(self, frame: ContextFrame) -> ContextFrame:
        return ContextFrame(
            kind=frame.kind,
            role=frame.role,
            payload=deepcopy(frame.payload),
            priority=frame.priority,
            stability=frame.stability,
            token_estimate=frame.token_estimate,
        )


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def partition_messages(
    messages: list[dict[str, Any]],
    *,
    protected_recent_turns: int,
    protected_tool_results: int,
    tail_token_budget: int | None = None,
) -> MessagePartitions:
    if protected_recent_turns <= 0:
        return MessagePartitions(compactable=deepcopy(messages), reserved=[])
    protected: list[dict[str, Any]] = []
    protected_tool_count = 0
    user_turns = 0
    protected_tokens = 0
    cutoff = len(messages)
    for message in reversed(messages):
        cutoff -= 1
        prepared = deepcopy(message)
        protected_tokens += _message_tokens(prepared)
        if prepared.get("role") == "tool":
            protected_tool_count += 1
            if protected_tool_count > protected_tool_results:
                prepared["content"] = "[旧 tool 输出已清理，关键结论见 handoff_memo 或 audit/events.jsonl]"
            protected.append(prepared)
            continue
        protected.append(prepared)
        if prepared.get("role") == "user":
            user_turns += 1
            if user_turns >= protected_recent_turns and (
                tail_token_budget is None or protected_tokens >= tail_token_budget
            ):
                break
    cutoff = _align_history_cutoff(messages, cutoff)
    protected = []
    protected_tool_count = 0
    for message in messages[cutoff:]:
        prepared = deepcopy(message)
        if prepared.get("role") == "tool":
            protected_tool_count += 1
            if protected_tool_count > protected_tool_results:
                prepared["content"] = "[旧 tool 输出已清理，关键结论见 handoff_memo 或 audit/events.jsonl]"
        protected.append(prepared)
    return MessagePartitions(compactable=deepcopy(messages[:cutoff]), reserved=protected)


def _message_tokens(message: dict[str, Any]) -> int:
    return estimate_tokens(json.dumps(message, ensure_ascii=False, sort_keys=True))


def _align_history_cutoff(messages: list[dict[str, Any]], cutoff: int) -> int:
    if cutoff <= 0 or cutoff >= len(messages):
        return cutoff
    compact_tail = messages[cutoff - 1]
    if compact_tail.get("role") == "tool":
        parent_cutoff = _find_parent_assistant_cutoff(messages, cutoff - 1)
        if parent_cutoff is not None:
            return parent_cutoff
    reserved_first = messages[cutoff]
    if reserved_first.get("role") != "tool":
        return cutoff
    tool_call_id = reserved_first.get("tool_call_id")
    parent_cutoff = _find_parent_assistant_cutoff(messages, cutoff, tool_call_id)
    return parent_cutoff if parent_cutoff is not None else cutoff


def _find_parent_assistant_cutoff(
    messages: list[dict[str, Any]],
    tool_message_index: int,
    tool_call_id: Any | None = None,
) -> int | None:
    if tool_call_id is None:
        tool_call_id = messages[tool_message_index].get("tool_call_id")
    for index in range(tool_message_index - 1, -1, -1):
        candidate = messages[index]
        if candidate.get("role") == "assistant" and _assistant_calls_tool(candidate, tool_call_id):
            return index
        if candidate.get("role") == "user":
            break
    return None


def _assistant_calls_tool(message: dict[str, Any], tool_call_id: Any) -> bool:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return False
    return any(isinstance(call, dict) and call.get("id") == tool_call_id for call in tool_calls)


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


def build_compaction_prompt(
    *,
    old_messages: list[dict[str, Any]],
    previous_handoff: str,
    scratchpad: dict[str, Any],
    source_refs: dict[str, str | None],
) -> str:
    return (
        "你是 coding-agent 的上下文压缩器和长期记忆提取器。\n"
        "你的任务有两个：\n"
        "1. 生成 handoff：给即将接手任务的下一个 LLM 实例使用，帮助它立刻继续当前任务。\n"
        "2. 生成 memories：从同一批材料中提取少量长期有复用价值的记忆。\n\n"
        "只输出一个 JSON 对象，不要输出 Markdown 代码块，不要输出解释文字。JSON 必须符合：\n"
        "{\n"
        '  "handoff": "string",\n'
        '  "memories": [\n'
        "    {\n"
        '      "type": "personal | procedural | knowledge",\n'
        '      "content": "string",\n'
        '      "confidence": 0.0,\n'
        '      "reason": "string"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "handoff 要求：\n"
        "- handoff 用 Markdown 写，且优先级高于 memories；不要为了生成 memories 牺牲 handoff 完整性。\n"
        "- 必须包含下面结构：\n\n"
        "## 目标\n"
        "[用户正在完成什么]\n\n"
        "## 约束与偏好\n"
        "[用户偏好、代码风格、限制条件、必须遵守的架构边界]\n\n"
        "## 进度\n"
        "### 已完成\n"
        "[已经完成的工作，包含具体文件、命令和结果]\n"
        "### 进行中\n"
        "[当前正在处理的事项]\n"
        "### 阻塞项\n"
        "[遇到的问题；没有则写 none]\n\n"
        "## 关键决策\n"
        "[重要技术决策以及原因]\n\n"
        "## 相关文件\n"
        "[读过、修改过或创建过的文件及简短说明]\n\n"
        "## 下一步\n"
        "[下一步应该做什么]\n\n"
        "## 关键上下文\n"
        "[错误信息、配置值、路径、测试结果等必须保留的细节]\n\n"
        "- 如果 previous_handoff 中已有摘要，请更新它：保留仍然有效的信息，移除过时信息，并合并 old_messages 的新增进展。\n"
        "- 如果 source_refs 中存在 archive/dialog 路径，在 handoff 中保留路径引用，方便后续 read_file 回溯。\n\n"
        "memories 要求：\n"
        "- 只记录长期有复用价值的信息。如果没有长期价值，返回空数组 []。\n"
        "- 不要记录普通执行流水、临时文件列表、一次性进度、普通工具调用结果、大段代码或日志。\n"
        "- 不要把 handoff 的内容机械复制进 memories。\n"
        "- 不要记录敏感信息、密钥、token 或个人隐私，除非用户明确要求保存。\n"
        "- 每条 memory 必须独立可读，不能写“如上”“本轮”等依赖上下文的表达。\n"
        "- content 要简洁，优先一句话；reason 说明为什么值得长期保存；confidence 必须是 0 到 1。\n"
        "- personal：用户稳定偏好、工作习惯、交流偏好、明确要求。\n"
        "- procedural：可复用做事方法、踩坑教训、测试/验证流程、下次应遵守的规则。\n"
        "- knowledge：项目稳定事实、架构决策、目录结构、技术约束。\n\n"
        "<source_refs>\n"
        f"{json.dumps(source_refs, ensure_ascii=False)}\n"
        "</source_refs>\n\n"
        "<previous_handoff>\n"
        f"{previous_handoff or 'none'}\n"
        "</previous_handoff>\n\n"
        "<scratchpad_json>\n"
        f"{json.dumps(scratchpad, ensure_ascii=False)}\n"
        "</scratchpad_json>\n\n"
        "<old_messages_json>\n"
        f"{json.dumps(old_messages, ensure_ascii=False)}\n"
        "</old_messages_json>"
    )


def build_handoff_prompt(
    *,
    old_messages: list[dict[str, Any]],
    previous_handoff: str,
    scratchpad: dict[str, Any],
    source_refs: dict[str, str | None],
) -> str:
    return build_compaction_prompt(
        old_messages=old_messages,
        previous_handoff=previous_handoff,
        scratchpad=scratchpad,
        source_refs=source_refs,
    )


def _format_compact_summary(summary_payload: dict[str, Any]) -> str:
    summary = summary_payload.get("summary")
    archive_path = summary_payload.get("archive_log_path")
    instruction = summary_payload.get("instruction")
    lines = ["[CONTEXT COMPACTION] Earlier turns were compacted."]
    if archive_path:
        lines.extend(["", f"Archive: {archive_path}"])
    if isinstance(summary, str) and summary.strip():
        lines.extend(["", summary.strip()])
    if isinstance(instruction, str) and instruction.strip():
        lines.extend(["", instruction.strip()])
    return "\n".join(lines)


def _prepend_handoff_sources(handoff: str, source_refs: dict[str, str | None]) -> str:
    lines = ["## Source References"]
    for key, value in source_refs.items():
        if value:
            lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append(handoff)
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
                    "",
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


def _clip(text: str, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    suffix = "truncated" if not label else f"{label} truncated"
    return text[:max_chars] + f"\n... [{suffix}]"


def _hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _usage_value(usage: Any, key: str) -> Any:
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int) or isinstance(value, dict):
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
