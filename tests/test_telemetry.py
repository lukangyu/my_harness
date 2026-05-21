import json

from coding_agent.telemetry import TelemetryLogger, build_workspace_snapshot


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_telemetry_writes_chinese_events_and_trace_jsonl(tmp_path):
    telemetry = TelemetryLogger(tmp_path / "logs", workspace_root=tmp_path)

    with telemetry.span("测试阶段", function="test_function", phase="test", metadata={"x": 1}):
        telemetry.event(
            "test.event",
            "这是一条中文事件",
            function="test_function",
            phase="test",
            metadata={"value": "ok"},
        )

    events = read_jsonl(tmp_path / "logs" / "events.jsonl")
    traces = read_jsonl(tmp_path / "logs" / "trace.jsonl")

    assert any(record["message_zh"] == "这是一条中文事件" for record in events)
    assert traces[0]["name"] == "测试阶段"
    assert traces[0]["function"] == "test_function"
    assert traces[0]["ok"] is True
    assert traces[0]["duration_ms"] >= 0


def test_workspace_snapshot_records_files_and_directories(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("print('a')", encoding="utf-8")
    (tmp_path / ".coding-agent").mkdir()
    (tmp_path / ".coding-agent" / "secret.json").write_text("ignored", encoding="utf-8")

    snapshot = build_workspace_snapshot(tmp_path)

    assert snapshot["exists"] is True
    assert "pkg" in snapshot["directories"]
    assert "pkg/a.py" in snapshot["files"]
    assert ".coding-agent/secret.json" not in snapshot["files"]


def test_workspace_snapshot_event_is_appended(tmp_path):
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    telemetry = TelemetryLogger(tmp_path / "logs", workspace_root=tmp_path)

    telemetry.workspace_snapshot(
        message_zh="记录 workspace 快照",
        function="test_workspace_snapshot_event_is_appended",
        phase="workspace",
    )

    events = read_jsonl(tmp_path / "logs" / "events.jsonl")
    assert events[0]["event"] == "workspace.snapshot"
    assert events[0]["message_zh"] == "记录 workspace 快照"
    assert "notes.txt" in events[0]["metadata"]["files"]
