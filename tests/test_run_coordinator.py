import json

import pytest

from coding_agent.agent import AgentResult
from coding_agent.interrupts import TaskInterrupted
from coding_agent.memory import MemoryStore
from coding_agent.run_coordinator import RunCoordinator
from coding_agent.run_store import RunStore
from coding_agent.session import SessionStore
from coding_agent.telemetry import TelemetryLogger


class FakeAgent:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def run(self, task, *, prior_messages, mode):
        self.calls.append({"task": task, "prior_messages": prior_messages, "mode": mode})
        if self.error is not None:
            raise self.error
        return self.result


def make_coordinator(tmp_path, task="inspect", mode="run"):
    run_store = RunStore(tmp_path)
    artifact, state = run_store.start_run(task=task, mode=mode, workspace_root=tmp_path)
    telemetry = TelemetryLogger(artifact.run_dir, workspace_root=tmp_path, run_id=artifact.run_id)
    memory_store = MemoryStore(
        tmp_path,
        dialog_dir=artifact.dialog_dir,
        tool_result_dir=artifact.tool_result_dir,
    )
    coordinator = RunCoordinator(
        run_store=run_store,
        run_artifact=artifact,
        task_state=state,
        telemetry=telemetry,
        memory_store=memory_store,
        session_store=SessionStore(tmp_path, telemetry=telemetry),
    )
    return coordinator, artifact


def test_run_coordinator_success_writes_session_state_and_report(tmp_path):
    coordinator, artifact = make_coordinator(tmp_path)
    agent = FakeAgent(
        AgentResult(
            final_answer="done",
            messages=[],
            conversation_messages=[{"role": "assistant", "content": "done"}],
            attempts=2,
            tool_steps=1,
            last_tool="read_file",
        )
    )

    result = coordinator.run(
        agent=agent,
        task="inspect",
        prior_messages=[],
        mode="run",
        show_cache_stats=True,
        workspace_root=tmp_path,
    )

    task_state = json.loads(artifact.task_state_path.read_text(encoding="utf-8"))
    report = json.loads(artifact.report_path.read_text(encoding="utf-8"))
    assert result.result.final_answer == "done"
    assert result.session_path.exists()
    assert task_state["status"] == "completed"
    assert task_state["attempts"] == 2
    assert task_state["tool_steps"] == 1
    assert report["session_path"] == str(result.session_path)
    assert report["status"] == "completed"
    assert agent.calls == [{"task": "inspect", "prior_messages": [], "mode": "run"}]


def test_run_coordinator_failure_writes_failed_state_and_report(tmp_path):
    coordinator, artifact = make_coordinator(tmp_path)
    agent = FakeAgent(error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        coordinator.run(
            agent=agent,
            task="inspect",
            prior_messages=[],
            mode="run",
            show_cache_stats=True,
            workspace_root=tmp_path,
        )

    task_state = json.loads(artifact.task_state_path.read_text(encoding="utf-8"))
    report = json.loads(artifact.report_path.read_text(encoding="utf-8"))
    assert task_state["status"] == "failed"
    assert task_state["stop_reason"] == "error"
    assert task_state["error"] == "boom"
    assert report["session_path"] is None
    assert report["status"] == "failed"


def test_run_coordinator_interruption_writes_interrupted_state_and_report(tmp_path):
    coordinator, artifact = make_coordinator(tmp_path)
    agent = FakeAgent(error=TaskInterrupted("user interrupted"))

    with pytest.raises(TaskInterrupted):
        coordinator.run(
            agent=agent,
            task="inspect",
            prior_messages=[],
            mode="run",
            show_cache_stats=True,
            workspace_root=tmp_path,
        )

    task_state = json.loads(artifact.task_state_path.read_text(encoding="utf-8"))
    report = json.loads(artifact.report_path.read_text(encoding="utf-8"))
    assert task_state["status"] == "interrupted"
    assert task_state["stop_reason"] == "user_interrupt"
    assert report["session_path"] is None
    assert report["status"] == "interrupted"
