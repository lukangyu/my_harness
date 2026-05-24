import json

from coding_agent.checkpoint.hook import CheckpointHook
from coding_agent.checkpoint.store import CheckpointStore
from coding_agent.execution.executor import ToolExecutor
from coding_agent.execution.policy import CommandPolicy
from coding_agent.execution.sandbox import WorkspaceSandbox
from coding_agent.execution.shell import ShellRunner
from coding_agent.execution.tools import create_default_tools
from coding_agent.orchestrator.lifecycle import AgentTurnContext
from coding_agent.session import ConversationStore


def make_executor(tmp_path):
    session = ConversationStore(tmp_path).start_conversation()
    sandbox = WorkspaceSandbox(tmp_path)
    store = CheckpointStore(
        conversation_dir=session.conversation_dir,
        workspace_root=sandbox.root,
        sandbox=sandbox,
    )
    shell = ShellRunner(CommandPolicy(allow=[], deny=[]), cwd=sandbox.root)
    tools = create_default_tools(sandbox, shell)
    executor = ToolExecutor(tools, lifecycle_hooks=[CheckpointHook(store)])
    ctx = AgentTurnContext(run_id="run-1", session_id=session.session_id)
    return executor, store, ctx


def tool_call(name, arguments):
    return {
        "id": "call-1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def tool_content(message):
    return json.loads(message["content"])


def test_read_file_records_last_seen_after_tool(tmp_path):
    executor, store, ctx = make_executor(tmp_path)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    result = executor.execute(tool_call("read_file", {"path": "notes.txt"}), ctx)

    assert tool_content(result)["ok"] is True
    record = store.load()["files"]["notes.txt"]
    assert record["exists"] is True
    assert record["source"] == "read_file"


def test_write_file_blocks_when_existing_file_was_not_read(tmp_path):
    executor, _store, ctx = make_executor(tmp_path)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    result = executor.execute(tool_call("write_file", {"path": "notes.txt", "content": "new"}), ctx)

    content = tool_content(result)
    assert content["ok"] is False
    assert content["code"] == "file_not_seen"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello"


def test_write_file_allows_new_file_without_last_seen(tmp_path):
    executor, store, ctx = make_executor(tmp_path)

    result = executor.execute(tool_call("write_file", {"path": "notes.txt", "content": "hello"}), ctx)

    assert tool_content(result)["ok"] is True
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello"
    assert store.load()["files"]["notes.txt"]["exists"] is True


def test_write_file_blocks_when_file_drifted_after_read(tmp_path):
    executor, _store, ctx = make_executor(tmp_path)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    executor.execute(tool_call("read_file", {"path": "notes.txt"}), ctx)
    (tmp_path / "notes.txt").write_text("external", encoding="utf-8")

    result = executor.execute(tool_call("write_file", {"path": "notes.txt", "content": "agent"}), ctx)

    content = tool_content(result)
    assert content["ok"] is False
    assert content["code"] == "file_drift_detected"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "external"


def test_apply_patch_update_blocks_when_target_drifted(tmp_path):
    executor, _store, ctx = make_executor(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    executor.execute(tool_call("read_file", {"path": "notes.txt"}), ctx)
    (tmp_path / "notes.txt").write_text("external\n", encoding="utf-8")
    patch = """*** Begin Patch
*** Update File: notes.txt
@@
-hello
+agent
*** End Patch"""

    result = executor.execute(tool_call("apply_patch", {"patch": patch}), ctx)

    assert tool_content(result)["code"] == "file_drift_detected"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "external\n"


def test_apply_patch_add_blocks_when_target_already_exists(tmp_path):
    executor, _store, ctx = make_executor(tmp_path)
    (tmp_path / "notes.txt").write_text("external\n", encoding="utf-8")
    patch = """*** Begin Patch
*** Add File: notes.txt
+agent
*** End Patch"""

    result = executor.execute(tool_call("apply_patch", {"patch": patch}), ctx)

    assert tool_content(result)["code"] == "file_already_exists"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "external\n"


def test_apply_patch_success_refreshes_checkpoint(tmp_path):
    executor, store, ctx = make_executor(tmp_path)
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    executor.execute(tool_call("read_file", {"path": "notes.txt"}), ctx)
    patch = """*** Begin Patch
*** Update File: notes.txt
@@
-hello
+agent
*** End Patch"""

    result = executor.execute(tool_call("apply_patch", {"patch": patch}), ctx)

    assert tool_content(result)["ok"] is True
    store.verify_file_not_drifted("notes.txt")
    assert store.load()["files"]["notes.txt"]["last_seen_run_id"] == "run-1"


def test_workspace_branch_change_blocks_write(tmp_path, monkeypatch):
    executor, store, ctx = make_executor(tmp_path)
    store.refresh_workspace()
    original = store.current_workspace_state

    def changed_workspace():
        state = original()
        return {**state, "branch": "other"}

    monkeypatch.setattr(store, "current_workspace_state", changed_workspace)

    result = executor.execute(tool_call("write_file", {"path": "notes.txt", "content": "hello"}), ctx)

    assert tool_content(result)["code"] == "workspace_drift_detected"
    assert not (tmp_path / "notes.txt").exists()
