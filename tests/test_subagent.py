import json
import threading

from coding_agent.context.context import WorkspaceContextOptions
from coding_agent.execution.policy import CommandPolicy
from coding_agent.execution.sandbox import WorkspaceSandbox
from coding_agent.execution.shell import ShellRunner
from coding_agent.execution.tools import ToolRegistry, create_default_tools, register_session_search_tool
from coding_agent.orchestrator.subagent import SubagentManager, register_subagent_tools


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.responses.pop(0)


class BlockingClient:
    def __init__(self, release_event):
        self.release_event = release_event

    def chat(self, messages, tools):
        self.release_event.wait(timeout=5)
        return {"message": {"role": "assistant", "content": "released"}}


def make_tools(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    shell = ShellRunner(CommandPolicy(allow=[], deny=[]), cwd=tmp_path)
    tools = create_default_tools(sandbox, shell)
    register_session_search_tool(tools, sandbox, [tmp_path / ".coding-agent"])
    return tools


def make_manager(tmp_path, client_factory):
    return SubagentManager(
        cwd=tmp_path,
        tools=make_tools(tmp_path),
        client_factory=client_factory,
        context_options=WorkspaceContextOptions(
            include_project_docs=False,
            include_file_tree=False,
            include_git_status=False,
            include_recent_commits=False,
        ),
        memory_store=None,
        telemetry=None,
        on_runtime_event=None,
        artifact_root=tmp_path / "subagents",
        max_steps=2,
    )


def test_subagent_tools_register_start_wait_cancel_schemas(tmp_path):
    manager = make_manager(
        tmp_path,
        lambda debug_dir: FakeClient([{"message": {"role": "assistant", "content": "done"}}]),
    )
    registry = ToolRegistry()

    register_subagent_tools(registry, manager)

    tool_names = [schema["function"]["name"] for schema in registry.schemas()]
    assert tool_names == ["start_subagent", "wait_subagent", "cancel_subagent"]


def test_subagent_manager_runs_read_only_agent_and_writes_report(tmp_path):
    manager = make_manager(
        tmp_path,
        lambda debug_dir: FakeClient([{"message": {"role": "assistant", "content": "sub report"}}]),
    )

    started = manager.start({"task": "inspect project", "context": "focus on tests"})
    waited = manager.wait({"subagent_id": started["subagent_id"], "timeout_seconds": 5})

    assert waited["ok"] is True
    assert waited["status"] == "completed"
    assert waited["summary"] == "sub report"
    report = json.loads((tmp_path / "subagents" / started["subagent_id"] / "report.json").read_text())
    assert report["status"] == "completed"
    assert (tmp_path / "subagents" / started["subagent_id"] / "messages.json").exists()


def test_subagent_manager_exposes_only_read_only_tools_to_child_agent(tmp_path):
    captured_clients = []

    def client_factory(debug_dir):
        client = FakeClient([{"message": {"role": "assistant", "content": "done"}}])
        captured_clients.append(client)
        return client

    manager = make_manager(tmp_path, client_factory)
    started = manager.start({"task": "inspect"})

    manager.wait({"subagent_id": started["subagent_id"], "timeout_seconds": 5})

    child_tool_names = {
        schema["function"]["name"]
        for schema in captured_clients[0].calls[0]["tools"]
    }
    assert {"list_files", "read_file", "search_text", "session_search", "run_shell"} <= child_tool_names
    assert "write_file" not in child_tool_names
    assert "apply_patch" not in child_tool_names
    assert "start_subagent" not in child_tool_names


def test_wait_subagent_times_out_without_losing_handle(tmp_path):
    release_event = threading.Event()
    manager = make_manager(tmp_path, lambda debug_dir: BlockingClient(release_event))
    started = manager.start({"task": "slow inspect"})

    first_wait = manager.wait({"subagent_id": started["subagent_id"], "timeout_seconds": 0})

    assert first_wait["status"] in {"pending", "running"}
    release_event.set()
    second_wait = manager.wait({"subagent_id": started["subagent_id"], "timeout_seconds": 5})
    assert second_wait["status"] == "completed"
