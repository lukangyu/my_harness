from __future__ import annotations

from pathlib import Path
from typing import Any

from coding_agent.agent import AgentLoop
from coding_agent.memory import MemoryStore
from coding_agent.run_result import RunTaskResult
from coding_agent.run_store import RunArtifact, RunStore, TaskState
from coding_agent.session import SessionStore
from coding_agent.telemetry import TelemetryLogger


class RunCoordinator:
    def __init__(
        self,
        *,
        run_store: RunStore,
        run_artifact: RunArtifact,
        task_state: TaskState,
        telemetry: TelemetryLogger,
        memory_store: MemoryStore,
        session_store: SessionStore,
    ) -> None:
        self.run_store = run_store
        self.run_artifact = run_artifact
        self.task_state = task_state
        self.telemetry = telemetry
        self.memory_store = memory_store
        self.session_store = session_store

    def run(
        self,
        *,
        agent: AgentLoop,
        task: str,
        prior_messages: list[dict[str, Any]] | None,
        mode: str,
        show_cache_stats: bool,
        workspace_root: Path,
    ) -> RunTaskResult:
        self.telemetry.event(
            "run.task.start",
            "RunCoordinator 收到任务并开始执行",
            function="RunCoordinator.run",
            phase="run",
            metadata={"mode": mode, "task_length": len(task), "has_prior_messages": bool(prior_messages)},
        )
        self.telemetry.workspace_snapshot(
            message_zh="任务开始前的 workspace 文件树和目录树快照",
            function="RunCoordinator.run",
            phase="workspace_before_task",
            root=workspace_root,
        )
        try:
            with self.telemetry.span(
                "执行完整任务",
                function="RunCoordinator.run",
                phase="run",
                metadata={"mode": mode},
            ):
                result = agent.run(task, prior_messages=prior_messages, mode=mode)
            self.telemetry.workspace_snapshot(
                message_zh="任务结束后的 workspace 文件树和目录树快照",
                function="RunCoordinator.run",
                phase="workspace_after_task",
                root=workspace_root,
            )
            session_path = self.session_store.save(result.conversation_messages)
            self.task_state.status = "completed"
            self.task_state.attempts = result.attempts
            self.task_state.tool_steps = result.tool_steps
            self.task_state.last_tool = result.last_tool
            self.task_state.stop_reason = result.stop_reason
            self.task_state.final_answer = result.final_answer
            self.run_store.write_task_state(self.run_artifact, self.task_state)
            self.run_store.write_report(
                self.run_artifact,
                self.task_state,
                session_path=session_path,
                usage=result.usage,
                files_changed=_memory_modified_files(self.memory_store),
            )
            self.telemetry.event(
                "run.task.end",
                "RunCoordinator 任务执行完成",
                function="RunCoordinator.run",
                phase="run",
                metadata={
                    "session_path": str(session_path),
                    "run_id": self.run_artifact.run_id,
                    "run_dir": str(self.run_artifact.run_dir),
                    "final_answer_length": len(result.final_answer),
                    "reached_max_steps": result.reached_max_steps,
                },
            )
            return RunTaskResult(result, session_path, show_cache_stats, self.run_artifact.run_id, self.run_artifact.run_dir)
        except Exception as exc:
            self.task_state.status = "failed"
            self.task_state.stop_reason = "error"
            self.task_state.error = str(exc)
            self.run_store.write_task_state(self.run_artifact, self.task_state)
            self.run_store.write_report(
                self.run_artifact,
                self.task_state,
                session_path=None,
                files_changed=_memory_modified_files(self.memory_store),
            )
            raise

    def record_progress(self, event: dict[str, Any]) -> None:
        if event.get("type") == "model_attempt":
            self.task_state.attempts = int(event.get("attempts") or self.task_state.attempts)
        if event.get("type") == "tool_step":
            self.task_state.tool_steps = int(event.get("tool_steps") or self.task_state.tool_steps)
            tool_name = event.get("tool")
            if isinstance(tool_name, str):
                self.task_state.last_tool = tool_name
        self.run_store.write_task_state(self.run_artifact, self.task_state)


def _memory_modified_files(memory_store: MemoryStore) -> list[str]:
    modified = memory_store.load_scratchpad().get("modified_files")
    return [path for path in modified if isinstance(path, str)] if isinstance(modified, list) else []
