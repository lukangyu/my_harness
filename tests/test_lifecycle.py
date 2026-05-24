import json

from coding_agent.context.context import WorkspaceContextOptions
from coding_agent.execution.executor import ToolExecutor
from coding_agent.execution.policy import CommandPolicy
from coding_agent.execution.sandbox import WorkspaceSandbox
from coding_agent.execution.shell import ShellRunner
from coding_agent.execution.tools import create_default_tools
from coding_agent.hooks.compaction_hook import ContextCompactionHook
from coding_agent.hooks.memory_hook import MemoryProjectionHook
from coding_agent.memory.store import MemoryStore
from coding_agent.orchestrator.agent_loop import AgentLoop
from coding_agent.orchestrator.lifecycle import AgentLifecycleHook, AgentTurnContext


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.responses.pop(0)


def make_tools(tmp_path):
    sandbox = WorkspaceSandbox(tmp_path)
    shell = ShellRunner(CommandPolicy(allow=[], deny=[]), cwd=tmp_path)
    return create_default_tools(sandbox, shell)


def context_options():
    return WorkspaceContextOptions(
        include_file_tree=False,
        include_git_status=False,
        include_recent_commits=False,
    )


def tool_call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_lifecycle_hook_can_modify_messages_before_llm(tmp_path):
    class AppendMessageHook(AgentLifecycleHook):
        def pre_llm(self, ctx: AgentTurnContext) -> None:
            ctx.messages.append({"role": "user", "content": "<dynamic_context>diff</dynamic_context>"})

    client = FakeClient([{"message": {"role": "assistant", "content": "answer"}}])
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
        lifecycle_hooks=[AppendMessageHook()],
    )

    result = agent.run("inspect")

    assert result.final_answer == "answer"
    assert client.calls[0]["messages"][-2] == {"role": "user", "content": "<dynamic_context>diff</dynamic_context>"}
    assert json.loads(client.calls[0]["messages"][-1]["content"])["kind"] == "current_task"


def test_pre_llm_hook_receives_context_entity_before_prompt_build(tmp_path):
    seen = []

    class ContextHook(AgentLifecycleHook):
        def pre_llm(self, ctx: AgentTurnContext) -> None:
            seen.append(ctx.context_entity.task)
            ctx.context_entity.add_summary_frame(
                {
                    "summary": "hook summary",
                    "archive_log_path": "dialog/archive.jsonl",
                    "instruction": "Use read_file if needed.",
                }
            )

    client = FakeClient([{"message": {"role": "assistant", "content": "answer"}}])
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
        lifecycle_hooks=[ContextHook()],
    )

    agent.run("inspect")

    assert seen == ["inspect"]
    assert client.calls[0]["messages"][2]["role"] == "assistant"
    assert client.calls[0]["messages"][2]["content"].startswith("[CONTEXT COMPACTION]")
    assert "hook summary" in client.calls[0]["messages"][2]["content"]
    assert json.loads(client.calls[0]["messages"][3]["content"])["kind"] == "current_task"


def test_context_compaction_hook_leaves_large_tool_frames_to_execution_offload(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    store = MemoryStore(tmp_path)
    hook = ContextCompactionHook(
        memory_store=store,
        compact_client=FakeClient([]),
    )
    client = FakeClient([{"message": {"role": "assistant", "content": "answer"}}])
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
        lifecycle_hooks=[hook],
    )
    prior = [{"role": "tool", "tool_call_id": "call-1", "name": "read_file", "content": "x" * 100}]

    agent.run("inspect", prior_messages=prior)

    tool_message = client.calls[0]["messages"][2]
    assert tool_message["content"] == "x" * 100
    assert not list(store.tool_result_dir.glob("*.txt"))


def test_context_compaction_hook_writes_structured_summary_to_handoff(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    store = MemoryStore(tmp_path)
    store.write_handoff("previous handoff")
    summary = (
        "## 目标\n继续任务\n\n"
        "## 约束与偏好\n保持工具层纯净\n\n"
        "## 进度\n### 已完成\n- 已归档旧消息\n\n"
        "## 下一步\n- 继续实现"
    )
    compact_client = FakeClient([{"message": {"role": "assistant", "content": summary}}])
    hook = ContextCompactionHook(memory_store=store, compact_client=compact_client)
    client = FakeClient([{"message": {"role": "assistant", "content": "answer"}}])
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=WorkspaceContextOptions(
            include_project_docs=False,
            include_file_tree=False,
            include_git_status=False,
            include_recent_commits=False,
            max_input_tokens=80,
            compact_threshold_ratio=0.1,
            protected_recent_turns=1,
        ),
        lifecycle_hooks=[hook],
    )
    prior = [
        {"role": "user", "content": "old " + "x" * 100},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new"},
    ]

    agent.run("inspect", prior_messages=prior)

    compact_prompt = compact_client.calls[0]["messages"][1]["content"]
    assert "## 目标" in compact_prompt
    assert "## 关键决策" in compact_prompt
    assert "previous handoff" in compact_prompt
    assert store.read_handoff().strip() == summary
    compacted_message = client.calls[0]["messages"][2]
    assert compacted_message["role"] == "assistant"
    assert compacted_message["content"].startswith("[CONTEXT COMPACTION]")
    assert "## 目标\n继续任务" in compacted_message["content"]


def test_lifecycle_hook_observes_llm_response(tmp_path):
    seen = []

    class CaptureResponseHook(AgentLifecycleHook):
        def after_llm(self, ctx: AgentTurnContext) -> None:
            seen.append(ctx.llm_response["message"]["content"])

    client = FakeClient([{"message": {"role": "assistant", "content": "answer"}}])
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
        lifecycle_hooks=[CaptureResponseHook()],
    )

    agent.run("inspect")

    assert seen == ["answer"]


def test_tool_executor_returns_clean_tool_message_and_runs_tool_hooks(tmp_path):
    events = []

    class ToolHook(AgentLifecycleHook):
        def pre_tool(self, ctx, tool_name, args):
            events.append(("pre", tool_name, dict(args)))

        def after_tool(self, ctx, tool_name, args, result):
            events.append(("after", tool_name, result.get("ok")))

    executor = ToolExecutor(make_tools(tmp_path), lifecycle_hooks=[ToolHook()])
    ctx = AgentTurnContext(turn_index=1)

    message = executor.execute(
        tool_call("write_file", json.dumps({"path": "notes.txt", "content": "hello"})),
        ctx,
    )

    assert message == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "write_file",
        "content": json.dumps(
            {"ok": True, "path": "notes.txt", "metadata": {"written_chars": 5}},
            ensure_ascii=False,
        ),
    }
    assert events == [
        ("pre", "write_file", {"path": "notes.txt", "content": "hello"}),
        ("after", "write_file", True),
    ]


def test_memory_projection_hook_records_tool_results_after_execution(tmp_path):
    store = MemoryStore(tmp_path)
    executor = ToolExecutor(
        make_tools(tmp_path),
        lifecycle_hooks=[MemoryProjectionHook(store)],
    )
    ctx = AgentTurnContext(turn_index=1)

    executor.execute(
        tool_call("write_file", json.dumps({"path": "notes.txt", "content": "hello"})),
        ctx,
    )

    scratchpad = store.load_scratchpad()
    assert scratchpad["modified_files"] == ["notes.txt"]
    assert store.load_file_summaries()["notes.txt"]["stale"] is True
    assert store.tool_index_path.exists()
