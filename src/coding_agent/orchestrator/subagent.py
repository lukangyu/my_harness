from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from coding_agent.context.assembler import ContextAssembler
from coding_agent.context.context import UsageStats, WorkspaceContextOptions
from coding_agent.context.prompt_builder import create_default_prompt_builder
from coding_agent.execution.tools import ToolRegistry
from coding_agent.interrupts import TaskInterrupted
from coding_agent.memory.store import MemoryStore
from coding_agent.orchestrator.agent_loop import AgentLoop, ChatClient
from coding_agent.orchestrator.lifecycle import AgentLifecycleHook, AgentLifecycleRegistry, AgentTurnContext
from coding_agent.runtime_events import RuntimeEvent
from coding_agent.telemetry.logger import TelemetryLogger


READ_ONLY_SUBAGENT_TOOLS = (
    "list_files",
    "read_file",
    "search_text",
    "session_search",
    "run_shell",
)


class ClientFactory(Protocol):
    def __call__(self, debug_dir: Path) -> ChatClient:
        ...


@dataclass
class SubagentJob:
    subagent_id: str
    task: str
    context: str
    artifact_dir: Path
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    status: str = "pending"
    result: dict[str, Any] | None = None
    error: str | None = None
    thread: threading.Thread | None = None


class SubagentManager:
    def __init__(
        self,
        *,
        cwd: Path,
        tools: ToolRegistry,
        client_factory: ClientFactory,
        context_options: WorkspaceContextOptions,
        memory_store: MemoryStore | None,
        telemetry: TelemetryLogger | None,
        on_runtime_event: Callable[[RuntimeEvent], None] | None,
        artifact_root: Path,
        max_steps: int,
    ) -> None:
        self.cwd = cwd
        self.tools = tools.subset(READ_ONLY_SUBAGENT_TOOLS)
        self.client_factory = client_factory
        self.context_options = context_options
        self.memory_store = memory_store
        self.telemetry = telemetry
        self.on_runtime_event = on_runtime_event
        self.artifact_root = artifact_root
        self.max_steps = max_steps
        self._jobs: dict[str, SubagentJob] = {}
        self._lock = threading.Lock()

    def start(self, arguments: dict[str, Any]) -> dict[str, Any]:
        task = _required_str(arguments, "task")
        context = str(arguments.get("context") or "")
        subagent_id = f"subagent-{uuid.uuid4().hex[:8]}"
        artifact_dir = self.artifact_root / subagent_id
        job = SubagentJob(
            subagent_id=subagent_id,
            task=task,
            context=context,
            artifact_dir=artifact_dir,
        )
        thread = threading.Thread(target=self._run_job, args=(job,), name=subagent_id, daemon=True)
        job.thread = thread
        with self._lock:
            self._jobs[subagent_id] = job
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self._write_report(job)
        self._event("subagent.start", "子代理任务已启动", job)
        thread.start()
        return {
            "ok": True,
            "subagent_id": subagent_id,
            "status": job.status,
            "artifact_path": str(artifact_dir),
        }

    def wait(self, arguments: dict[str, Any]) -> dict[str, Any]:
        job = self._get_job(arguments)
        timeout_seconds = _optional_timeout(arguments.get("timeout_seconds"), default=30.0)
        job.done_event.wait(timeout=timeout_seconds)
        return self._job_response(job)

    def cancel(self, arguments: dict[str, Any]) -> dict[str, Any]:
        job = self._get_job(arguments)
        if job.done_event.is_set():
            return self._job_response(job)
        job.cancel_event.set()
        if job.status in {"pending", "running"}:
            job.status = "cancellation_requested"
            self._write_report(job)
        self._event("subagent.cancel", "子代理任务请求取消", job)
        return self._job_response(job)

    def _run_job(self, job: SubagentJob) -> None:
        job.status = "running"
        self._write_report(job)
        try:
            prompt_builder = create_default_prompt_builder()
            prompt_builder.modify_system_prompt(
                "system_base",
                (
                    "You are a read-only subagent for coding-agent. "
                    "Investigate the delegated task, use only read/search/verification tools, "
                    "do not modify files, and return a concise structured report."
                ),
            )
            lifecycle_registry = AgentLifecycleRegistry()
            lifecycle_registry.add("on_turn_start", _SubagentCancelHook(job.cancel_event), order=0)
            lifecycle_registry.add("pre_tool", _SubagentCancelHook(job.cancel_event), order=0)
            agent = AgentLoop(
                client=self.client_factory(job.artifact_dir / "debug"),
                tools=self.tools,
                max_steps=self.max_steps,
                cwd=self.cwd,
                context_options=self.context_options,
                telemetry=self.telemetry,
                memory_store=self.memory_store,
                lifecycle_registry=lifecycle_registry,
                prompt_builder=prompt_builder,
                context_assembler=ContextAssembler(
                    cwd=self.cwd,
                    options=self.context_options,
                    memory_store=self.memory_store,
                ),
                on_runtime_event=self._subagent_runtime_event(job),
            )
            result = agent.run(_task_with_context(job), prior_messages=[], mode="subagent")
            payload = {
                "ok": True,
                "subagent_id": job.subagent_id,
                "status": "completed",
                "summary": result.final_answer,
                "findings": [],
                "risks": [],
                "artifact_path": str(job.artifact_dir),
                "attempts": result.attempts,
                "tool_steps": result.tool_steps,
                "usage": _usage_to_dict(result.usage),
            }
            _write_json(job.artifact_dir / "messages.json", {"messages": result.messages})
            job.status = "completed"
            job.result = payload
        except TaskInterrupted as exc:
            job.status = "cancelled"
            job.error = str(exc) or "subagent cancelled"
            job.result = self._job_response(job)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.result = self._job_response(job)
        finally:
            self._write_report(job)
            job.done_event.set()
            self._event("subagent.end", "子代理任务结束", job)

    def _get_job(self, arguments: dict[str, Any]) -> SubagentJob:
        subagent_id = _required_str(arguments, "subagent_id")
        with self._lock:
            job = self._jobs.get(subagent_id)
        if job is None:
            raise ValueError(f"Unknown subagent_id: {subagent_id}")
        return job

    def _job_response(self, job: SubagentJob) -> dict[str, Any]:
        if job.result is not None:
            return dict(job.result)
        return {
            "ok": job.status not in {"failed"},
            "subagent_id": job.subagent_id,
            "status": job.status,
            "summary": "",
            "findings": [],
            "risks": [],
            "artifact_path": str(job.artifact_dir),
            "attempts": None,
            "tool_steps": None,
            "usage": None,
            "error": job.error,
        }

    def _write_report(self, job: SubagentJob) -> None:
        payload = {
            "schema_version": 1,
            "subagent_id": job.subagent_id,
            "task": job.task,
            "status": job.status,
            "result": job.result,
            "error": job.error,
            "updated_at": _now(),
        }
        _write_json(job.artifact_dir / "report.json", payload)

    def _subagent_runtime_event(self, job: SubagentJob) -> Callable[[RuntimeEvent], None]:
        def emit(event: RuntimeEvent) -> None:
            self._runtime_event(
                RuntimeEvent(
                    type="subagent.event",
                    message=event.message,
                    metadata={"subagent_id": job.subagent_id, "inner": event.metadata, "inner_type": event.type},
                )
            )

        return emit

    def _event(self, event_type: str, message: str, job: SubagentJob) -> None:
        metadata = {"subagent_id": job.subagent_id, "status": job.status, "artifact_path": str(job.artifact_dir)}
        if self.telemetry is not None:
            self.telemetry.event(event_type, message, function="SubagentManager", phase="subagent", metadata=metadata)
        self._runtime_event(RuntimeEvent(type=event_type, message=message, metadata=metadata))

    def _runtime_event(self, event: RuntimeEvent) -> None:
        if self.on_runtime_event is not None:
            self.on_runtime_event(event)


class _SubagentCancelHook(AgentLifecycleHook):
    def __init__(self, cancel_event: threading.Event) -> None:
        self.cancel_event = cancel_event

    def on_turn_start(self, ctx: AgentTurnContext) -> None:
        self._raise_if_cancelled()

    def pre_tool(self, ctx: AgentTurnContext, tool_name: str, args: dict[str, Any]) -> None:
        self._raise_if_cancelled()

    def _raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise TaskInterrupted("subagent cancelled")


def register_subagent_tools(registry: ToolRegistry, manager: SubagentManager) -> None:
    registry.register(
        "start_subagent",
        "Start a read-only subagent task and return a current-run handle.",
        _parameters(
            properties={
                "task": {"type": "string"},
                "context": {"type": "string"},
            },
            required=["task"],
        ),
        manager.start,
    )
    registry.register(
        "wait_subagent",
        "Wait for a read-only subagent task by handle.",
        _parameters(
            properties={
                "subagent_id": {"type": "string"},
                "timeout_seconds": {"type": "number", "default": 30},
            },
            required=["subagent_id"],
        ),
        manager.wait,
    )
    registry.register(
        "cancel_subagent",
        "Request cooperative cancellation for a read-only subagent task.",
        _parameters(
            properties={"subagent_id": {"type": "string"}},
            required=["subagent_id"],
        ),
        manager.cancel,
    )


def _parameters(*, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _required_str(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_timeout(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        raise ValueError("timeout_seconds must be a number")
    return max(0.0, timeout)


def _task_with_context(job: SubagentJob) -> str:
    if not job.context.strip():
        return job.task
    return f"{job.task}\n\nAdditional context from parent agent:\n{job.context.strip()}"


def _usage_to_dict(usage: UsageStats | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cached_tokens": usage.cached_tokens,
        "cache_hit_ratio": usage.cache_hit_ratio,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
