import json

from coding_agent.memory.store import MemoryStore


def test_memory_store_records_file_tool_facts(tmp_path):
    store = MemoryStore(tmp_path)

    store.record_tool_result(
        tool="write_file",
        arguments={"path": "notes.txt", "content": "hello"},
        result={"ok": True, "path": "notes.txt"},
    )
    store.record_tool_result(
        tool="read_file",
        arguments={"path": "notes.txt"},
        result={"ok": True, "path": "notes.txt", "content": "hello"},
    )

    scratchpad = store.load_scratchpad()
    assert scratchpad["modified_files"] == ["notes.txt"]
    assert scratchpad["read_files"] == ["notes.txt"]
    assert store.tool_index_path.exists()


def test_memory_store_records_shell_failure_as_known_issue(tmp_path):
    store = MemoryStore(tmp_path)

    store.record_tool_result(
        tool="run_shell",
        arguments={"command": "pytest"},
        result={"ok": False, "command": "pytest", "exit_code": 1, "stderr": "failed"},
    )

    scratchpad = store.load_scratchpad()
    assert scratchpad["last_verified_commands"] == [
        {"command": "pytest", "ok": False, "exit_code": 1, "timed_out": None}
    ]
    assert scratchpad["known_issues"][0]["error"] == "failed"


def test_memory_anchor_renders_json_block(tmp_path):
    store = MemoryStore(tmp_path)
    store.save_scratchpad({"project_goal": "优化上下文", "modified_files": ["a.py"]})

    anchor = store.render_memory_anchor(max_chars=1000)

    assert anchor.startswith("<memory_anchor>")
    assert "优化上下文" in anchor
    assert json.loads(anchor.removeprefix("<memory_anchor>\n").removesuffix("\n</memory_anchor>"))["modified_files"] == ["a.py"]


def test_memory_store_updates_and_renders_valid_file_summary(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("import json\n\nclass Worker:\n    pass\n", encoding="utf-8")
    store = MemoryStore(tmp_path)

    summary = store.update_file_summary("module.py")
    rendered = store.render_file_summaries(
        candidate_paths=["module.py"],
        max_count=8,
        max_chars=4000,
    )

    assert summary is not None
    assert "<file_summaries>" in rendered
    assert 'path="module.py"' in rendered
    assert "class Worker" in rendered


def test_memory_store_does_not_render_stale_or_hash_mismatched_summary(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("class Worker:\n    pass\n", encoding="utf-8")
    store = MemoryStore(tmp_path)
    store.update_file_summary("module.py")

    path.write_text("class Changed:\n    pass\n", encoding="utf-8")
    rendered = store.render_file_summaries(
        candidate_paths=["module.py"],
        max_count=8,
        max_chars=4000,
    )
    summaries = store.load_file_summaries()

    assert rendered == ""
    assert summaries["module.py"]["stale"] is True


def test_memory_store_invalidates_summary_after_write_and_patch_results(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("class Worker:\n    pass\n", encoding="utf-8")
    store = MemoryStore(tmp_path)
    store.update_file_summary("module.py")

    store.record_tool_result(
        tool="write_file",
        arguments={"path": "module.py"},
        result={"ok": True, "path": "module.py"},
    )
    assert store.load_file_summaries()["module.py"]["stale_reason"] == "written"

    path.write_text("class Worker:\n    pass\n", encoding="utf-8")
    store.update_file_summary("module.py")
    store.record_tool_result(
        tool="apply_patch",
        arguments={"patch": "..."},
        result={"ok": True, "changed_files": ["module.py"]},
    )
    assert store.load_file_summaries()["module.py"]["stale_reason"] == "patched"


def test_memory_store_archives_dialog_messages_as_jsonl(tmp_path):
    store = MemoryStore(tmp_path)

    path = store.archive_dialog_messages(
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "answer"},
        ]
    )

    assert path is not None
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"role": "user", "content": "old"}
    assert path.parent == store.dialog_dir


def test_memory_store_offloads_long_tool_result(tmp_path):
    store = MemoryStore(tmp_path)

    result = store.offload_tool_result(tool="read_file", content="x" * 5000, max_inline_chars=100)

    assert result["offloaded"] is True
    assert result["original_chars"] == 5000
    assert "完整 tool 输出已转存" in result["content"]
    assert len(result["content"]) < 300
    assert (tmp_path / result["path"]).read_text(encoding="utf-8") == "x" * 5000


def test_memory_store_keeps_short_tool_result_inline(tmp_path):
    store = MemoryStore(tmp_path)

    result = store.offload_tool_result(tool="read_file", content="short", max_inline_chars=100)

    assert result == {"content": "short", "offloaded": False}
