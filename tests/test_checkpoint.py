import json

import pytest

from coding_agent.checkpoint.patch_targets import parse_patch_targets
from coding_agent.checkpoint.store import CheckpointConflict, CheckpointStore
from coding_agent.execution.sandbox import WorkspaceSandbox
from coding_agent.session import ConversationStore


def make_store(tmp_path):
    session = ConversationStore(tmp_path).start_conversation()
    sandbox = WorkspaceSandbox(tmp_path)
    return CheckpointStore(
        conversation_dir=session.conversation_dir,
        workspace_root=sandbox.root,
        sandbox=sandbox,
    ), session


def test_checkpoint_store_records_workspace_identity(tmp_path):
    store, _session = make_store(tmp_path)

    store.refresh_workspace(run_id="run-1", session_id="session-1")

    payload = store.load()
    assert payload["version"] == 1
    assert payload["workspace"]["repo_root"] == str(tmp_path.resolve())
    assert "fingerprint" in payload["workspace"]
    assert payload["workspace"]["last_seen_run_id"] == "run-1"
    assert payload["files"] == {}


def test_checkpoint_store_upserts_file_record(tmp_path):
    store, _session = make_store(tmp_path)
    (tmp_path / "notes.txt").write_text("old", encoding="utf-8")

    store.record_file("notes.txt", source="read_file", run_id="run-1", session_id="session-1")
    first = store.load()["files"]["notes.txt"]
    (tmp_path / "notes.txt").write_text("new", encoding="utf-8")
    store.record_file("notes.txt", source="read_file", run_id="run-2", session_id="session-2")

    second = store.load()["files"]["notes.txt"]
    assert second["exists"] is True
    assert second["content_hash"] != first["content_hash"]
    assert second["last_seen_run_id"] == "run-2"
    assert list(store.load()["files"]) == ["notes.txt"]


def test_checkpoint_store_records_deleted_tombstone(tmp_path):
    store, _session = make_store(tmp_path)

    store.record_deleted("missing.txt", source="apply_patch", run_id="run-1", session_id="session-1")

    record = store.load()["files"]["missing.txt"]
    assert record["exists"] is False
    assert record["content_hash"] is None
    assert record["source"] == "apply_patch"


def test_checkpoint_file_conflict_when_existing_file_was_not_read(tmp_path):
    store, _session = make_store(tmp_path)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    with pytest.raises(CheckpointConflict) as excinfo:
        store.verify_file_not_drifted("notes.txt")

    assert excinfo.value.result["code"] == "file_not_seen"
    assert excinfo.value.result["path"] == "notes.txt"


def test_checkpoint_file_conflict_when_file_drifted_after_read(tmp_path):
    store, _session = make_store(tmp_path)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    store.record_file("notes.txt", source="read_file")
    (tmp_path / "notes.txt").write_text("changed", encoding="utf-8")

    with pytest.raises(CheckpointConflict) as excinfo:
        store.verify_file_not_drifted("notes.txt")

    assert excinfo.value.result["code"] == "file_drift_detected"
    assert "Re-read this file" in excinfo.value.result["instruction"]


def test_checkpoint_survives_session_rotation(tmp_path):
    conversation_store = ConversationStore(tmp_path)
    session = conversation_store.start_conversation()
    sandbox = WorkspaceSandbox(tmp_path)
    store = CheckpointStore(
        conversation_dir=session.conversation_dir,
        workspace_root=sandbox.root,
        sandbox=sandbox,
    )
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    store.record_file("notes.txt", source="read_file")

    next_session = conversation_store.compact_session(
        session,
        seed_messages=[{"role": "assistant", "content": "[CONTEXT COMPACTION] summary"}],
        summary="summary",
    )

    assert next_session.conversation_dir == session.conversation_dir
    checkpoint_path = session.conversation_dir / "checkpoints" / "checkpoint.json"
    assert checkpoint_path.exists()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert "notes.txt" in payload["files"]


def test_parse_patch_targets_detects_operations():
    patch = """*** Begin Patch
*** Add File: new.txt
+hello
*** Update File: old.txt
*** Move to: moved.txt
@@
-old
+new
*** Delete File: delete.txt
*** End Patch"""

    targets = parse_patch_targets(patch)

    assert [(target.path, target.operation) for target in targets] == [
        ("new.txt", "add"),
        ("old.txt", "move_source"),
        ("moved.txt", "move_target"),
        ("delete.txt", "delete"),
    ]
