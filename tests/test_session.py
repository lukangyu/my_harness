import json

import pytest

from coding_agent.session import ConversationStore, LegacySessionError


def test_start_conversation_creates_active_session_directory(tmp_path):
    store = ConversationStore(tmp_path)

    session = store.start_conversation()

    assert session.conversation_dir == tmp_path / ".coding-agent" / "conversations" / session.conversation_id
    assert session.session_dir == session.conversation_dir / "sessions" / session.session_id
    assert session.session_path.exists()
    assert session.memory_dir.exists()
    assert session.runs_dir.exists()
    conversation = json.loads(session.conversation_path.read_text(encoding="utf-8"))
    session_payload = json.loads(session.session_path.read_text(encoding="utf-8"))
    assert conversation["active_session_id"] == session.session_id
    assert conversation["sessions"][0]["status"] == "active"
    assert session_payload["status"] == "active"
    assert session_payload["messages"] == []


def test_save_and_load_session_messages(tmp_path):
    store = ConversationStore(tmp_path)
    session = store.start_conversation()
    messages = [{"role": "user", "content": "hello"}]

    store.save_session_messages(session, messages)

    assert store.load_session_messages(session) == messages


def test_resume_conversation_directory_uses_active_session(tmp_path):
    store = ConversationStore(tmp_path)
    session = store.start_conversation()

    resumed = store.resume(session.conversation_dir)

    assert resumed.conversation_id == session.conversation_id
    assert resumed.session_id == session.session_id


def test_resume_session_directory_loads_that_session(tmp_path):
    store = ConversationStore(tmp_path)
    session = store.start_conversation()

    resumed = store.resume(session.session_dir)

    assert resumed.conversation_id == session.conversation_id
    assert resumed.session_id == session.session_id


def test_resume_legacy_flat_session_json_returns_clear_error(tmp_path):
    legacy_dir = tmp_path / ".coding-agent" / "sessions"
    legacy_dir.mkdir(parents=True)
    legacy_path = legacy_dir / "old.json"
    legacy_path.write_text("[]", encoding="utf-8")

    with pytest.raises(LegacySessionError, match="legacy flat session files are no longer supported"):
        ConversationStore(tmp_path).resume(legacy_path)


def test_compact_session_marks_old_session_and_creates_next_active_session(tmp_path):
    store = ConversationStore(tmp_path)
    session = store.start_conversation()
    store.save_session_messages(session, [{"role": "user", "content": "old"}])

    next_session = store.compact_session(
        session,
        seed_messages=[{"role": "assistant", "content": "[CONTEXT COMPACTION] summary"}],
        summary="summary",
    )

    old_payload = json.loads(session.session_path.read_text(encoding="utf-8"))
    new_payload = json.loads(next_session.session_path.read_text(encoding="utf-8"))
    conversation = json.loads(session.conversation_path.read_text(encoding="utf-8"))
    assert old_payload["status"] == "compacted"
    assert old_payload["next_session_id"] == next_session.session_id
    assert old_payload["messages"] == [{"role": "user", "content": "old"}]
    assert new_payload["status"] == "active"
    assert new_payload["previous_session_id"] == session.session_id
    assert new_payload["messages"] == [{"role": "assistant", "content": "[CONTEXT COMPACTION] summary"}]
    assert conversation["active_session_id"] == next_session.session_id


def test_compact_session_carries_forward_scratchpad(tmp_path):
    store = ConversationStore(tmp_path)
    session = store.start_conversation()
    scratchpad = session.memory_dir / "scratchpad.json"
    scratchpad.write_text('{"active_todos":["refactor session"]}', encoding="utf-8")

    next_session = store.compact_session(
        session,
        seed_messages=[{"role": "assistant", "content": "summary"}],
        summary="handoff summary",
    )

    assert (next_session.memory_dir / "scratchpad.json").read_text(encoding="utf-8") == scratchpad.read_text(
        encoding="utf-8"
    )
    assert (next_session.memory_dir / "handoff.md").read_text(encoding="utf-8") == "handoff summary"


def test_compact_session_switches_active_session_last(tmp_path, monkeypatch):
    store = ConversationStore(tmp_path)
    session = store.start_conversation()
    original_write = store._atomic_write_json

    def fail_before_conversation_switch(path, payload):
        if path == session.conversation_path and payload.get("active_session_id") != session.session_id:
            raise RuntimeError("boom")
        original_write(path, payload)

    monkeypatch.setattr(store, "_atomic_write_json", fail_before_conversation_switch)

    with pytest.raises(RuntimeError, match="boom"):
        store.compact_session(
            session,
            seed_messages=[{"role": "assistant", "content": "summary"}],
            summary="handoff summary",
        )

    conversation = json.loads(session.conversation_path.read_text(encoding="utf-8"))
    assert conversation["active_session_id"] == session.session_id
    assert store.resume(session.conversation_dir).session_id == session.session_id


def test_resume_ignores_orphan_new_session_when_active_not_switched(tmp_path):
    store = ConversationStore(tmp_path)
    session = store.start_conversation()
    orphan_dir = session.conversation_dir / "sessions" / "orphan"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "session.json").write_text(
        json.dumps(
            {
                "conversation_id": session.conversation_id,
                "session_id": "orphan",
                "status": "active",
                "previous_session_id": session.session_id,
                "next_session_id": None,
                "created_at": "2026-05-24T00:00:00",
                "updated_at": "2026-05-24T00:00:00",
                "compacted_at": None,
                "messages": [],
            }
        ),
        encoding="utf-8",
    )

    resumed = store.resume(session.conversation_dir)

    assert resumed.session_id == session.session_id
