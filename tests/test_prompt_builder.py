import json

import pytest

from coding_agent.context.context import Context, ContextFrame, WorkspaceContextOptions
from coding_agent.context.prompt_builder import (
    PromptBuilder,
    create_default_prompt_builder,
)


def test_prompt_builder_layers_are_injected_in_stable_order(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(include_project_docs=False, include_file_tree=False),
        task="task",
    )
    builder = (
        PromptBuilder()
        .add_system_base("base")
        .add_tool_guidance("guidance")
        .add_tool_enforcement("enforcement")
    )

    system = builder.build_final_messages(context)[0]["content"]

    assert system.index("base") < system.index("guidance") < system.index("enforcement")


def test_modify_system_prompt_updates_named_layer(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(include_project_docs=False, include_file_tree=False),
        task="task",
    )
    builder = PromptBuilder().add_system_base("base").add_tool_guidance("old")

    builder.modify_system_prompt("tool_guidance", "new")

    system = builder.build_final_messages(context)[0]["content"]
    assert "new" in system
    assert "old" not in system


def test_prompt_state_cannot_be_mutated_from_outside():
    builder = PromptBuilder().add_system_base("base")
    state = builder.state

    with pytest.raises(AttributeError):
        state.layers += (state.layers[0],)

    assert len(builder.state.layers) == 1


def test_build_final_messages_outputs_kv_cache_order(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(include_project_docs=False, include_file_tree=False),
        task="current task",
        mode="chat",
    )
    context.update_workspace_snapshot(
        context.workspace_snapshot.with_updates(project_docs={"README.md": "hello"})
    )
    context.add_message({"role": "user", "content": "earlier"})

    messages = create_default_prompt_builder().build_final_messages(context)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert json.loads(messages[1]["content"])["kind"] == "context"
    assert messages[2] == {"role": "user", "content": "earlier"}
    assert messages[3]["role"] == "user"
    assert json.loads(messages[3]["content"]) == {
        "kind": "current_task",
        "mode": "chat",
        "content": "current task",
    }


def test_default_prompt_builder_warns_when_memory_tools_are_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(include_project_docs=False, include_file_tree=False),
        task="task",
        tool_schemas=[{"type": "function", "function": {"name": "read_file"}}],
    )

    system = create_default_prompt_builder().build_final_messages(context)[0]["content"]

    assert "session_search tool is currently offline" in system
    assert "memory tool is currently offline" in system


def test_prompt_builder_keeps_tool_definitions_out_of_system_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(include_project_docs=False, include_file_tree=False),
        task="task",
        tool_schemas=[
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "run_shell"}},
        ],
    )

    system = create_default_prompt_builder().build_final_messages(context)[0]["content"]

    assert "AVAILABLE TOOL NAMES" not in system
    assert "- read_file" not in system
    assert "- run_shell" not in system
    assert "Use the provided tool schemas as the sole source of available tools" in system
    assert "When asked what tools are available" in system
    assert "multi_tool_use" not in system


def test_tool_availability_warning_is_omitted_when_tool_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(include_project_docs=False, include_file_tree=False),
        task="task",
        tool_schemas=[
            {"type": "function", "function": {"name": "memory"}},
            {"type": "function", "function": {"name": "session_search"}},
        ],
    )

    system = create_default_prompt_builder().build_final_messages(context)[0]["content"]

    assert "currently offline" not in system


def test_prompt_builder_mentions_context_archives_when_read_file_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(include_project_docs=False, include_file_tree=False),
        task="task",
        tool_schemas=[{"type": "function", "function": {"name": "read_file"}}],
    )

    system = create_default_prompt_builder().build_final_messages(context)[0]["content"]

    assert "CONTEXT MANAGEMENT NOTICE" in system
    assert "agent_context/tool_result/" in system
    assert "agent_context/dialog/" in system
    assert "use your read_file tool" in system


def test_prompt_builder_can_wrap_context_as_xml_later(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent.resolve()))
    context = Context(
        cwd=tmp_path,
        options=WorkspaceContextOptions(include_project_docs=False, include_file_tree=False),
        task="task",
    )

    messages = PromptBuilder(context_format="xml").add_system_base("base").build_final_messages(context)

    assert messages[1]["content"].startswith("<context_json>")
    assert messages[-1]["content"].startswith("<current_task_json>")
