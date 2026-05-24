import json

from coding_agent.context.context import WorkspaceContextOptions
from coding_agent.execution.executor import ToolExecutor
from coding_agent.execution.policy import CommandPolicy
from coding_agent.execution.sandbox import WorkspaceSandbox
from coding_agent.execution.shell import ShellRunner
from coding_agent.execution.tools import create_default_tools
from coding_agent.hooks.compaction_hook import ContextCompactionHook
from coding_agent.hooks.memory_hook import MemoryProjectionHook
from coding_agent.hooks.memory_search_hook import MemorySearchHook
from coding_agent.hooks.tool_result_offload_hook import ToolResultOffloadHook
from coding_agent.memory.store import MemoryStore
from coding_agent.orchestrator.agent_loop import AgentLoop
from coding_agent.orchestrator.lifecycle import AgentLifecycleBus, AgentLifecycleHook, AgentLifecycleRegistry, AgentTurnContext
from coding_agent.session import ConversationStore, SessionRuntime


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


def lifecycle_registry(phase: str, hook: AgentLifecycleHook, *, order: int = 1) -> AgentLifecycleRegistry:
    registry = AgentLifecycleRegistry()
    registry.add(phase, hook, order=order)
    return registry


def xml_json_payload(content: str, tag: str) -> dict:
    return json.loads(content.removeprefix(f"<{tag}>\n").removesuffix(f"\n</{tag}>"))


def test_lifecycle_registry_orders_hooks_within_each_phase():
    events = []

    class Hook(AgentLifecycleHook):
        def __init__(self, name):
            self.name = name

        def pre_llm(self, ctx):
            events.append(("pre_llm", self.name))

        def pre_tool(self, ctx, tool_name, args):
            events.append(("pre_tool", self.name))

    registry = AgentLifecycleRegistry()
    registry.add("pre_llm", Hook("late-pre-llm"), order=20)
    registry.add("pre_llm", Hook("early-pre-llm"), order=1)
    registry.add("pre_tool", Hook("early-pre-tool"), order=1)
    registry.add("pre_tool", Hook("late-pre-tool"), order=20)
    bus = AgentLifecycleBus(registry)

    bus.pre_llm(AgentTurnContext())
    bus.pre_tool(AgentTurnContext(), "read_file", {})

    assert events == [
        ("pre_llm", "early-pre-llm"),
        ("pre_llm", "late-pre-llm"),
        ("pre_tool", "early-pre-tool"),
        ("pre_tool", "late-pre-tool"),
    ]


def test_after_tool_pipeline_passes_transformed_result_to_later_hooks():
    seen = []

    class TransformHook(AgentLifecycleHook):
        def after_tool(self, ctx, tool_name, args, result):
            return {**result, "transformed": True}

    class ObserveHook(AgentLifecycleHook):
        def after_tool(self, ctx, tool_name, args, result):
            seen.append(dict(result))

    registry = AgentLifecycleRegistry()
    registry.add("after_tool", TransformHook(), order=1)
    registry.add("after_tool", ObserveHook(), order=2)
    result = AgentLifecycleBus(registry).after_tool(AgentTurnContext(), "tool", {}, {"ok": True})

    assert result == {"ok": True, "transformed": True}
    assert seen == [{"ok": True, "transformed": True}]


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
        lifecycle_registry=lifecycle_registry("pre_llm", AppendMessageHook()),
    )

    result = agent.run("inspect")

    assert result.final_answer == "answer"
    assert client.calls[0]["messages"][-2] == {"role": "user", "content": "inspect"}
    assert client.calls[0]["messages"][-1] == {"role": "user", "content": "<dynamic_context>diff</dynamic_context>"}


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
        lifecycle_registry=lifecycle_registry("pre_llm", ContextHook()),
    )

    agent.run("inspect")

    assert seen == ["inspect"]
    assert client.calls[0]["messages"][3]["tool_calls"][0]["function"]["name"] == "sync_compacted_context"
    compacted_payload = xml_json_payload(client.calls[0]["messages"][4]["content"], "compacted_context")
    assert "hook summary" in compacted_payload["compact_summary"]
    assert client.calls[0]["messages"][-1] == {"role": "user", "content": "inspect"}


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
        lifecycle_registry=lifecycle_registry("pre_llm", hook),
    )
    prior = [{"role": "tool", "tool_call_id": "call-1", "name": "read_file", "content": "x" * 100}]

    agent.run("inspect", prior_messages=prior)

    tool_message = client.calls[0]["messages"][3]
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
        lifecycle_registry=lifecycle_registry("pre_llm", hook),
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
    assert client.calls[0]["messages"][3]["tool_calls"][0]["function"]["name"] == "sync_compacted_context"
    compacted_payload = xml_json_payload(client.calls[0]["messages"][4]["content"], "compacted_context")
    assert "## 目标\n继续任务" in compacted_payload["compact_summary"]


def test_context_compaction_hook_writes_long_term_memories_from_json_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    conversation_memory_dir = tmp_path / ".coding-agent" / "conversations" / "c1" / "memory"
    store = MemoryStore(tmp_path, conversation_memory_dir=conversation_memory_dir)
    compact_client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "handoff": "## 目标\n继续任务\n\n## 下一步\n继续实现",
                            "memories": [
                                {
                                    "type": "procedural",
                                    "content": "修改压缩 prompt 后需要同步生命周期测试。",
                                    "confidence": 0.8,
                                    "reason": "本轮测试覆盖了压缩输出格式。",
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                }
            }
        ]
    )
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
        lifecycle_registry=lifecycle_registry("pre_llm", hook),
    )

    agent.run(
        "inspect",
        prior_messages=[
            {"role": "user", "content": "old " + "x" * 100},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new"},
        ],
    )

    raw_files = list((conversation_memory_dir / "raw").glob("*.jsonl"))
    assert store.read_handoff().strip() == "## 目标\n继续任务\n\n## 下一步\n继续实现"
    assert len(raw_files) == 1
    record = json.loads(raw_files[0].read_text(encoding="utf-8").strip())
    assert record["type"] == "procedural"
    assert record["content"] == "修改压缩 prompt 后需要同步生命周期测试。"
    assert record["source"] == "context_compaction"
    assert record["evidence"] == [record["evidence"][0]]
    assert record["evidence"][0].endswith(".jsonl")


def test_context_compaction_falls_back_to_handoff_only_when_json_parse_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    conversation_memory_dir = tmp_path / ".coding-agent" / "conversations" / "c1" / "memory"
    store = MemoryStore(tmp_path, conversation_memory_dir=conversation_memory_dir)
    compact_client = FakeClient([{"message": {"role": "assistant", "content": "## 目标\n继续任务"}}])
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
        lifecycle_registry=lifecycle_registry("pre_llm", hook),
    )

    agent.run(
        "inspect",
        prior_messages=[
            {"role": "user", "content": "old " + "x" * 100},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new"},
        ],
    )

    assert store.read_handoff().strip() == "## 目标\n继续任务"
    assert not (conversation_memory_dir / "raw").exists()


def test_context_compaction_hook_rotates_session_and_carries_scratchpad(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    conversation_store = ConversationStore(tmp_path)
    session = conversation_store.start_conversation()
    (session.memory_dir / "scratchpad.json").write_text('{"active_todos":["next file"]}', encoding="utf-8")
    store = MemoryStore(tmp_path, memory_dir=session.memory_dir)
    compact_client = FakeClient([{"message": {"role": "assistant", "content": "## 目标\n继续任务"}}])
    hook = ContextCompactionHook(memory_store=store, compact_client=compact_client)
    client = FakeClient([{"message": {"role": "assistant", "content": "answer"}}])
    runtime = SessionRuntime(conversation_store, session)
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
        lifecycle_registry=lifecycle_registry("pre_llm", hook),
        session_runtime=runtime,
    )
    prior = [
        {"role": "user", "content": "old " + "x" * 100},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new"},
    ]

    agent.run("inspect", prior_messages=prior)

    old_payload = json.loads(session.session_path.read_text(encoding="utf-8"))
    new_payload = json.loads(runtime.current.session_path.read_text(encoding="utf-8"))
    assert runtime.current.session_id != session.session_id
    assert old_payload["status"] == "compacted"
    assert new_payload["messages"] == [{"role": "user", "content": "new"}]
    assert json.loads((runtime.current.memory_dir / "scratchpad.json").read_text(encoding="utf-8"))[
        "active_todos"
    ] == ["next file"]
    assert (runtime.current.memory_dir / "handoff.md").read_text(encoding="utf-8").strip() == "## 目标\n继续任务"


def test_memory_search_hook_injects_synthetic_pair_after_current_user(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    store = MemoryStore(tmp_path)
    store.append_long_term_memories(
        [
            {
                "type": "procedural",
                "content": "修改 prompt 组装后需要同步 prompt_builder 测试。",
                "confidence": 0.9,
                "reason": "测试直接约束消息顺序。",
            }
        ],
        source="test",
        evidence=["events.jsonl"],
    )
    client = FakeClient([{"message": {"role": "assistant", "content": "answer"}}])
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
        lifecycle_registry=lifecycle_registry("pre_llm", MemorySearchHook(store)),
    )

    result = agent.run("prompt 组装怎么改")

    messages = client.calls[0]["messages"]
    assert result.final_answer == "answer"
    assert messages[-3] == {"role": "user", "content": "prompt 组装怎么改"}
    assert messages[-2]["tool_calls"][0]["function"]["name"] == "auto_memory_search"
    assert messages[-1]["name"] == "auto_memory_search"
    memory_payload = xml_json_payload(messages[-1]["content"], "memory_search_results")
    assert memory_payload["results"][0]["content"] == "修改 prompt 组装后需要同步 prompt_builder 测试。"
    assert result.conversation_messages == [
        {"role": "user", "content": "prompt 组装怎么改"},
        {"role": "assistant", "content": "answer"},
    ]


def test_internal_virtual_tool_schemas_are_sent_but_not_user_registered(tmp_path):
    client = FakeClient([{"message": {"role": "assistant", "content": "answer"}}])
    tools = make_tools(tmp_path)
    agent = AgentLoop(
        client,
        tools,
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
    )

    agent.run("inspect")

    request_tool_names = {
        schema["function"]["name"]
        for schema in client.calls[0]["tools"]
        if isinstance(schema.get("function"), dict)
    }
    registered_tool_names = {
        schema["function"]["name"]
        for schema in tools.schemas()
        if isinstance(schema.get("function"), dict)
    }
    assert "sync_session_context" in request_tool_names
    assert "auto_memory_search" in request_tool_names
    assert "sync_session_context" not in registered_tool_names
    assert "auto_memory_search" not in registered_tool_names


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
        lifecycle_registry=lifecycle_registry("after_llm", CaptureResponseHook()),
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

    registry = AgentLifecycleRegistry()
    tool_hook = ToolHook()
    registry.add("pre_tool", tool_hook, order=1)
    registry.add("after_tool", tool_hook, order=1)
    executor = ToolExecutor(make_tools(tmp_path), lifecycle=AgentLifecycleBus(registry))
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
        lifecycle=AgentLifecycleBus(lifecycle_registry("after_tool", MemoryProjectionHook(store))),
    )
    ctx = AgentTurnContext(turn_index=1)

    executor.execute(
        tool_call("write_file", json.dumps({"path": "notes.txt", "content": "hello"})),
        ctx,
    )

    scratchpad = store.load_scratchpad()
    assert "notes.txt" in scratchpad["modified_files"]
    assert not (store.memory_dir / "file_summaries.json").exists()
    assert not (store.memory_dir / "tool_index.jsonl").exists()


def test_tool_result_offload_hook_transforms_large_tool_message_after_memory_projection(tmp_path):
    store = MemoryStore(tmp_path)
    (tmp_path / "large.txt").write_text("x" * 5000, encoding="utf-8")
    registry = AgentLifecycleRegistry()
    registry.add("after_tool", MemoryProjectionHook(store), order=1)
    registry.add("after_tool", ToolResultOffloadHook(store, max_inline_chars=200), order=90)
    executor = ToolExecutor(make_tools(tmp_path), lifecycle=AgentLifecycleBus(registry))

    message = executor.execute(
        tool_call("read_file", json.dumps({"path": "large.txt", "max_chars": 5000})),
        AgentTurnContext(turn_index=1),
    )

    content = json.loads(message["content"])
    scratchpad = store.load_scratchpad()
    assert content["offloaded"] is True
    assert content["path"].endswith(".txt")
    assert "Full tool output is archived" in content["instruction"]
    assert (tmp_path / content["path"]).exists()
    assert scratchpad["read_files"] == ["large.txt"]
