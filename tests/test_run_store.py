import json

from coding_agent.context.context import UsageStats
from coding_agent.telemetry.store import RunStore


def test_run_store_creates_run_artifacts_and_initial_task_state(tmp_path):
    store = RunStore(tmp_path)

    artifact, state = store.start_run(task="do work", mode="run", workspace_root=tmp_path)

    assert artifact.run_dir.exists()
    assert artifact.debug_dir.exists()
    assert artifact.dialog_dir.exists()
    assert artifact.tool_result_dir.exists()
    task_state = json.loads(artifact.task_state_path.read_text(encoding="utf-8"))
    assert task_state["schema_version"] == 1
    assert task_state["run_id"] == artifact.run_id
    assert task_state["task"] == "do work"
    assert task_state["status"] == "running"
    assert state.run_id == artifact.run_id


def test_run_store_writes_report_with_usage_and_artifact_paths(tmp_path):
    store = RunStore(tmp_path)
    artifact, state = store.start_run(task="do work", mode="chat", workspace_root=tmp_path)
    state.status = "completed"
    state.stop_reason = "final_answer"
    state.final_answer = "done"
    state.attempts = 2
    state.tool_steps = 3

    store.write_task_state(artifact, state)
    store.write_report(
        artifact,
        state,
        session_path=tmp_path / ".coding-agent" / "sessions" / "s.json",
        usage=UsageStats(input_tokens=100, output_tokens=10, cached_tokens=50),
        files_changed=["a.py"],
    )

    report = json.loads(artifact.report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["run_id"] == artifact.run_id
    assert report["status"] == "completed"
    assert report["usage"]["cache_hit_ratio"] == 0.5
    assert report["files_changed"] == ["a.py"]
    assert report["artifact_paths"]["trace"].endswith("trace.jsonl")
    assert store.index_path.exists()
