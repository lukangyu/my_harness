import json
import os
import time
from pathlib import Path

import httpx

from coding_agent.context.context import UsageStats, WorkspaceContextOptions
from coding_agent.execution.policy import CommandPolicy
from coding_agent.execution.sandbox import WorkspaceSandbox
from coding_agent.execution.shell import ShellRunner
from coding_agent.execution.tools import ToolRegistry, create_default_tools
from coding_agent.orchestrator.agent_loop import AgentLoop
from coding_agent.runtime_events import RuntimeEvent
from coding_agent.llm import LLMError, OpenAICompatibleClient
from coding_agent.session import SessionStore


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


def test_agent_runs_tool_then_returns_final_answer(tmp_path):
    client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call(
                            "write_file",
                            json.dumps({"path": "notes.txt", "content": "hello"}),
                        )
                    ],
                },
                "usage": UsageStats(input_tokens=100, output_tokens=5, cached_tokens=40),
            },
            {
                "message": {"role": "assistant", "content": "done"},
                "usage": UsageStats(input_tokens=120, output_tokens=8, cached_tokens=90),
            },
        ]
    )
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=3,
        cwd=tmp_path,
        context_options=context_options(),
    )

    result = agent.run("write a note")

    assert result.final_answer == "done"
    assert result.usage == UsageStats(input_tokens=120, output_tokens=8, cached_tokens=90)
    assert result.reached_max_steps is False
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello"
    assert client.calls[0]["messages"][0]["role"] == "system"
    assert "You are coding-agent" in client.calls[0]["messages"][0]["content"]
    assert client.calls[0]["messages"][1]["tool_calls"][0]["function"]["name"] == "sync_session_context"
    assert client.calls[0]["messages"][2]["name"] == "sync_session_context"
    assert client.calls[0]["messages"][2]["content"].startswith("<session_context>")
    assert client.calls[0]["messages"][3] == {"role": "user", "content": "write a note"}
    assert client.calls[1]["messages"][-1]["role"] == "tool"
    assert client.calls[1]["messages"][-1]["tool_call_id"] == "call_1"
    assert client.calls[1]["messages"][-1]["name"] == "write_file"
    assert json.loads(client.calls[1]["messages"][-1]["content"]) == {
        "ok": True,
        "path": "notes.txt",
        "metadata": {"written_chars": 5},
    }
    assert result.conversation_messages == [
        {"role": "user", "content": "write a note"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                tool_call(
                    "write_file",
                    json.dumps({"path": "notes.txt", "content": "hello"}),
                )
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "write_file",
            "content": json.dumps(
                {"ok": True, "path": "notes.txt", "metadata": {"written_chars": 5}},
                ensure_ascii=False,
            ),
        },
        {"role": "assistant", "content": "done"},
    ]


def test_agent_treats_model_requested_virtual_tool_as_unknown(tmp_path):
    client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("sync_session_context", "{}")],
                }
            },
            {"message": {"role": "assistant", "content": "done"}},
        ]
    )
    agent = AgentLoop(client, make_tools(tmp_path), max_steps=2, cwd=tmp_path)

    agent.run("inspect")

    tool_message = client.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["name"] == "sync_session_context"
    assert json.loads(tool_message["content"]) == {
        "ok": False,
        "error": "Unknown tool: sync_session_context",
    }


def test_agent_executes_parallel_safe_tool_calls_concurrently_and_preserves_order(tmp_path):
    registry = ToolRegistry()
    completions = []

    def slow_read(arguments):
        time.sleep(0.2)
        completions.append("read_file")
        return {"ok": True, "path": "slow.txt", "content": "slow"}

    def fast_search(arguments):
        completions.append("search_text")
        return {"ok": True, "matches": []}

    registry.register("read_file", "Read", {"type": "object", "properties": {}}, slow_read)
    registry.register("search_text", "Search", {"type": "object", "properties": {}}, fast_search)
    client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call("read_file", "{}"),
                        tool_call("search_text", "{}", call_id="call_2"),
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "done"}},
        ]
    )
    agent = AgentLoop(client, registry, max_steps=2, cwd=tmp_path)

    agent.run("inspect")

    tool_messages = [
        message
        for message in client.calls[1]["messages"]
        if message.get("role") == "tool" and message.get("tool_call_id") in {"call_1", "call_2"}
    ]
    assert completions == ["search_text", "read_file"]
    assert [message["name"] for message in tool_messages] == ["read_file", "search_text"]
    assert [message["tool_call_id"] for message in tool_messages] == ["call_1", "call_2"]


def test_agent_runs_mixed_write_batch_serially(tmp_path):
    registry = ToolRegistry()
    events = []

    def write_file(arguments):
        events.append("write_start")
        time.sleep(0.05)
        events.append("write_end")
        return {"ok": True, "path": "notes.txt"}

    def read_file(arguments):
        events.append("read_start")
        events.append("read_end")
        return {"ok": True, "path": "notes.txt", "content": "hello"}

    registry.register("write_file", "Write", {"type": "object", "properties": {}}, write_file)
    registry.register("read_file", "Read", {"type": "object", "properties": {}}, read_file)
    client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call("write_file", "{}"),
                        tool_call("read_file", "{}", call_id="call_2"),
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "done"}},
        ]
    )
    agent = AgentLoop(client, registry, max_steps=2, cwd=tmp_path)

    agent.run("inspect")

    assert events == ["write_start", "write_end", "read_start", "read_end"]


def test_agent_waits_for_multiple_subagents_in_parallel(tmp_path):
    registry = ToolRegistry()
    completions = []

    def wait_subagent(arguments):
        if arguments["subagent_id"] == "a":
            time.sleep(0.2)
        completions.append(arguments["subagent_id"])
        return {"ok": True, "subagent_id": arguments["subagent_id"], "status": "completed"}

    registry.register("wait_subagent", "Wait", {"type": "object", "properties": {}}, wait_subagent)
    client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call("wait_subagent", json.dumps({"subagent_id": "a"})),
                        tool_call("wait_subagent", json.dumps({"subagent_id": "b"}), call_id="call_2"),
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "done"}},
        ]
    )
    agent = AgentLoop(client, registry, max_steps=2, cwd=tmp_path)

    agent.run("inspect")

    tool_messages = [
        message
        for message in client.calls[1]["messages"]
        if message.get("role") == "tool" and message.get("tool_call_id") in {"call_1", "call_2"}
    ]
    assert completions == ["b", "a"]
    assert [json.loads(message["content"])["subagent_id"] for message in tool_messages] == ["a", "b"]


def test_agent_preserves_previous_usage_when_final_response_has_no_usage(tmp_path):
    first_usage = UsageStats(input_tokens=100, output_tokens=5, cached_tokens=40)
    client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call(
                            "write_file",
                            json.dumps({"path": "notes.txt", "content": "hello"}),
                        )
                    ],
                },
                "usage": first_usage,
            },
            {
                "message": {"role": "assistant", "content": "done"},
            },
        ]
    )
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=3,
        cwd=tmp_path,
        context_options=context_options(),
    )

    result = agent.run("write a note")

    assert result.final_answer == "done"
    assert result.usage == first_usage


def test_agent_reports_invalid_json_tool_arguments(tmp_path):
    client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("write_file", "{not json}")],
                }
            },
            {"message": {"role": "assistant", "content": "fixed"}},
        ]
    )
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=3,
        cwd=tmp_path,
        context_options=context_options(),
    )

    result = agent.run("write a note")

    tool_message = client.calls[1]["messages"][-1]
    tool_result = json.loads(tool_message["content"])
    assert result.final_answer == "fixed"
    assert tool_result["ok"] is False
    assert "invalid json" in tool_result["error"].lower()


def test_agent_emits_tool_call_event_before_execution(tmp_path):
    events = []
    client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call(
                            "write_file",
                            json.dumps({"path": "notes.txt", "content": "hello"}),
                        )
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "done"}},
        ]
    )
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=3,
        cwd=tmp_path,
        context_options=context_options(),
        on_tool_call=events.append,
    )

    result = agent.run("write a note")

    assert result.final_answer == "done"
    assert events == [
        tool_call(
            "write_file",
            json.dumps({"path": "notes.txt", "content": "hello"}),
        )
    ]


def test_agent_emits_progress_events_after_model_and_tool_steps(tmp_path):
    progress = []
    client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call(
                            "write_file",
                            json.dumps({"path": "notes.txt", "content": "hello"}),
                        )
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "done"}},
        ]
    )
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=3,
        cwd=tmp_path,
        context_options=context_options(),
        on_progress=progress.append,
    )

    result = agent.run("write a note")

    assert result.final_answer == "done"
    assert progress == [
        {"type": "model_attempt", "attempts": 1, "step": 1},
        {"type": "tool_step", "tool_steps": 1, "tool": "write_file", "step": 1},
        {"type": "model_attempt", "attempts": 2, "step": 2},
    ]


def test_agent_emits_runtime_event_for_tool_result(tmp_path):
    events: list[RuntimeEvent] = []
    client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("run_shell", json.dumps({"command": "git log --oneline -n 8"}))],
                }
            },
            {"message": {"role": "assistant", "content": "done"}},
        ]
    )
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=3,
        cwd=tmp_path,
        context_options=context_options(),
        on_runtime_event=events.append,
    )

    agent.run("show recent commits")

    tool_events = [event for event in events if event.type == "tool.result"]
    assert tool_events == [
        RuntimeEvent(
            type="tool.result",
            message="工具 run_shell 执行完成",
            metadata={
                "tool": "run_shell",
                "ok": False,
                "error": "Command not in allow list",
                "exit_code": None,
                "timed_out": False,
            },
        )
    ]


def test_agent_stops_after_max_steps(tmp_path):
    client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("list_files", "{}")],
                },
                "usage": UsageStats(input_tokens=80, output_tokens=12, cached_tokens=20),
            }
        ]
    )
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
    )

    result = agent.run("inspect files")

    assert result.reached_max_steps is True
    assert "max steps" in result.final_answer.lower()
    assert result.usage == UsageStats(input_tokens=80, output_tokens=12, cached_tokens=20)


def test_agent_places_prior_messages_after_workspace_context(tmp_path):
    client = FakeClient([{"message": {"role": "assistant", "content": "answer"}}])
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
    )
    prior_messages = [{"role": "user", "content": "earlier"}]

    result = agent.run("new task", prior_messages=prior_messages)

    assert result.final_answer == "answer"
    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "You are coding-agent" in messages[0]["content"]
    assert messages[1]["tool_calls"][0]["function"]["name"] == "sync_session_context"
    assert messages[2]["name"] == "sync_session_context"
    assert messages[3] == {"role": "user", "content": "earlier"}
    assert messages[4] == {"role": "user", "content": "new task"}


def test_agent_result_conversation_messages_exclude_generated_prompt_blocks(tmp_path):
    client = FakeClient([{"message": {"role": "assistant", "content": "answer"}}])
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
    )
    prior_messages = [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "old answer"},
    ]

    result = agent.run("new task", prior_messages=prior_messages, mode="chat")

    assert result.conversation_messages == [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new task"},
        {"role": "assistant", "content": "answer"},
    ]
    serialized = json.dumps(result.conversation_messages)
    assert "<coding_agent_prefix" not in serialized
    assert "<workspace_context>" not in serialized
    assert "<current_task>" not in serialized


def test_agent_sanitizes_model_echoed_prompt_markers_from_conversation_messages(tmp_path):
    client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "saw <workspace_context>secret</workspace_context> and <current_task>x</current_task>",
                    "reasoning_content": "prefix <coding_agent_prefix version='1'>hidden",
                }
            }
        ]
    )
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
    )

    result = agent.run("new task", mode="chat")

    serialized = json.dumps(result.conversation_messages)
    assert "<workspace_context>" not in serialized
    assert "<current_task>" not in serialized
    assert "<coding_agent_prefix" not in serialized
    assert "generated workspace context omitted" in serialized


def test_agent_reuses_sanitized_conversation_without_duplicating_prompt_blocks(tmp_path):
    first_client = FakeClient([{"message": {"role": "assistant", "content": "first"}}])
    first_agent = AgentLoop(
        first_client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
    )
    first_result = first_agent.run("first task", mode="chat")

    second_client = FakeClient([{"message": {"role": "assistant", "content": "second"}}])
    second_agent = AgentLoop(
        second_client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
    )

    second_agent.run(
        "second task",
        prior_messages=first_result.conversation_messages,
        mode="chat",
    )

    second_messages = second_client.calls[0]["messages"]
    rendered = json.dumps(second_messages)
    assert rendered.count("You are coding-agent") == 1
    assert second_messages[1]["tool_calls"][0]["function"]["name"] == "sync_session_context"
    assert second_messages[2]["name"] == "sync_session_context"
    assert second_messages[3:] == [
        {"role": "user", "content": "first task"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "second task"},
    ]


def test_agent_includes_workspace_context_for_empty_prior_messages(tmp_path):
    (tmp_path / "README.md").write_text("hello workspace", encoding="utf-8")
    client = FakeClient([{"message": {"role": "assistant", "content": "answer"}}])
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
    )

    result = agent.run("new task", prior_messages=[])

    assert result.final_answer == "answer"
    assert client.calls[0]["messages"][0]["role"] == "system"
    assert "You are coding-agent" in client.calls[0]["messages"][0]["content"]
    assert client.calls[0]["messages"][1]["tool_calls"][0]["function"]["name"] == "sync_session_context"
    assert client.calls[0]["messages"][2]["name"] == "sync_session_context"
    session_payload = json.loads(
        client.calls[0]["messages"][2]["content"].removeprefix("<session_context>\n").removesuffix("\n</session_context>")
    )
    assert session_payload["kind"] == "session_context"
    assert client.calls[0]["messages"][3] == {"role": "user", "content": "new task"}


def test_agent_emits_runtime_event_for_prompt_context(tmp_path):
    events: list[RuntimeEvent] = []
    (tmp_path / "README.md").write_text("hello workspace", encoding="utf-8")
    client = FakeClient([{"message": {"role": "assistant", "content": "answer"}}])
    agent = AgentLoop(
        client,
        make_tools(tmp_path),
        max_steps=1,
        cwd=tmp_path,
        context_options=context_options(),
        on_runtime_event=events.append,
    )

    agent.run("new task")

    assert events == [
        RuntimeEvent(
            type="context.built",
            message="上下文已组装",
            metadata={
                "tool_count": 6,
                "recent_messages": 0,
                "memory_anchor": False,
                "handoff_memo": False,
            },
        )
    ]


def test_llm_client_posts_openai_compatible_chat_request(monkeypatch):
    requests = []

    def fake_post(url, *, headers, json, timeout):
        requests.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 30,
                    "prompt_tokens_details": {"cached_tokens": 70},
                },
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OpenAICompatibleClient("https://example.test/v1/", "secret", "model-a", timeout=12.5)

    result = client.chat([{"role": "user", "content": "hi"}], [{"type": "function"}])

    assert result == {
        "message": {"role": "assistant", "content": "hello"},
        "usage": UsageStats(input_tokens=100, output_tokens=30, cached_tokens=70),
    }
    assert requests == [
        {
            "url": "https://example.test/v1/chat/completions",
            "headers": {"Authorization": "Bearer secret"},
            "json": {
                "model": "model-a",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "tools": [{"type": "function"}],
                "tool_choice": "auto",
            },
            "timeout": 12.5,
        }
    ]


def test_llm_client_writes_debug_log_for_chat_request(monkeypatch, tmp_path):
    def fake_post(url, *, headers, json, timeout):
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [{"message": {"role": "assistant", "content": "hello"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OpenAICompatibleClient(
        "https://example.test/v1/",
        "secret",
        "model-a",
        debug_dir=tmp_path,
    )

    client.chat([{"role": "user", "content": "hi"}], [])

    logs = list(tmp_path.glob("*.json"))
    assert len(logs) == 1
    record = json.loads(logs[0].read_text(encoding="utf-8"))
    assert record["request"]["url"] == "https://example.test/v1/chat/completions"
    assert record["request"]["payload"] == {
        "model": "model-a",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    assert record["response"]["json"]["choices"][0]["message"]["content"] == "hello"
    assert record["response"]["parsed"]["message"]["content"] == "hello"
    assert "secret" not in logs[0].read_text(encoding="utf-8")
    assert "Authorization" not in logs[0].read_text(encoding="utf-8")


def test_llm_client_reports_empty_non_streaming_response(monkeypatch):
    def fake_post(url, *, headers, json, timeout):
        return httpx.Response(200, request=httpx.Request("POST", url), content=b"")

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OpenAICompatibleClient("https://example.test/v1/", "secret", "model-a")

    try:
        client.chat([{"role": "user", "content": "hi"}], [])
    except LLMError as exc:
        assert "non-streaming model response" in str(exc)
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected LLMError")


def test_llm_client_streams_openai_compatible_chat_request(monkeypatch):
    requests = []

    class FakeStreamResponse:
        def __init__(self, url):
            self.url = url
            self.status_checked = False

        def raise_for_status(self):
            self.status_checked = True

        def iter_lines(self):
            return iter(
                [
                    'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"thinking "}}]}',
                    'data: {"choices":[{"delta":{"content":"hel"}}]}',
                    'data: {"choices":[{"delta":{"content":"lo"}}]}',
                    "data: [DONE]",
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_stream(method, url, *, headers, json, timeout):
        requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeStreamResponse(url)

    monkeypatch.setattr(httpx, "stream", fake_stream)
    client = OpenAICompatibleClient(
        "https://example.test/v1/",
        "secret",
        "model-a",
        timeout=12.5,
        stream=True,
    )

    result = client.chat([{"role": "user", "content": "hi"}], [{"type": "function"}])

    assert result == {
        "message": {"role": "assistant", "content": "hello", "reasoning_content": "thinking "},
        "usage": None,
    }
    assert requests == [
        {
            "method": "POST",
            "url": "https://example.test/v1/chat/completions",
            "headers": {"Authorization": "Bearer secret"},
            "json": {
                "model": "model-a",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "tools": [{"type": "function"}],
                "tool_choice": "auto",
            },
            "timeout": 12.5,
        }
    ]


def test_llm_client_preserves_reasoning_content_when_stream_has_no_content(monkeypatch):
    class FakeStreamResponse:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(
                [
                    'data: {"choices":[{"delta":{"role":"assistant","content":null,"reasoning_content":"分析"}}]}',
                    'data: {"choices":[{"delta":{"content":null,"reasoning_content":"一下项目"}}]}',
                    "data: [DONE]",
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(httpx, "stream", lambda *args, **kwargs: FakeStreamResponse())
    client = OpenAICompatibleClient(
        "https://example.test/v1/",
        "secret",
        "model-a",
        timeout=12.5,
        stream=True,
    )

    result = client.chat([{"role": "user", "content": "hi"}], [])

    assert result == {
        "message": {"role": "assistant", "content": "", "reasoning_content": "分析一下项目"},
        "usage": None,
    }


def test_llm_client_accumulates_streamed_tool_calls(monkeypatch):
    class FakeStreamResponse:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(
                [
                    'data: {"choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":""}}]}}]}',
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\":\\"README.md\\"}"}}]}}]}',
                    'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(httpx, "stream", lambda *args, **kwargs: FakeStreamResponse())
    client = OpenAICompatibleClient(
        "https://example.test/v1/",
        "secret",
        "model-a",
        timeout=12.5,
        stream=True,
    )

    result = client.chat([{"role": "user", "content": "hi"}], [])

    assert result == {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                }
            ],
        },
        "usage": None,
    }


def test_llm_client_raises_llm_error_for_httpx_errors(monkeypatch):
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OpenAICompatibleClient("https://example.test", "secret", "model-a")

    try:
        client.chat([], [])
    except LLMError as exc:
        assert "no route" in str(exc)
    else:
        raise AssertionError("expected LLMError")


def test_session_store_saves_records_under_project_sessions(tmp_path, monkeypatch):
    class FixedDateTime:
        @classmethod
        def now(cls):
            return cls()

        def strftime(self, fmt):
            return "20260513-120000-123456"

    import coding_agent.session as session_module

    monkeypatch.setattr(session_module, "datetime", FixedDateTime)
    records = [{"role": "user", "content": "hello"}]

    path = SessionStore(tmp_path).save(records)

    assert path == Path(tmp_path / ".coding-agent" / "sessions" / "20260513-120000-123456.json")
    assert path.read_text(encoding="utf-8") == json.dumps(records, indent=2, ensure_ascii=False)


def test_session_store_loads_saved_records(tmp_path):
    records = [{"role": "user", "content": "hello"}]
    path = SessionStore(tmp_path).save(records)

    assert SessionStore.load(path) == records


def test_session_store_load_rejects_non_list_json(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"role": "user"}), encoding="utf-8")

    try:
        SessionStore.load(path)
    except ValueError as exc:
        assert "list" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_session_store_load_rejects_non_dict_item(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps([{"role": "user", "content": "hello"}, "bad"]), encoding="utf-8")

    try:
        SessionStore.load(path)
    except ValueError as exc:
        assert "dict" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_session_store_load_rejects_system_and_generated_prompt_blocks(tmp_path):
    for records in [
        [{"role": "system", "content": "do not restore"}],
        [{"role": "user", "content": "<workspace_context>\nsecret\n</workspace_context>"}],
        [{"role": "assistant", "content": "<coding_agent_prefix version='1'>"}],
        [{"role": "user", "content": "<current_task>\nold task\n</current_task>"}],
    ]:
        path = tmp_path / "session.json"
        path.write_text(json.dumps(records), encoding="utf-8")

        try:
            SessionStore.load(path)
        except ValueError as exc:
            assert "session" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError")


def test_session_store_load_rejects_tool_content_with_generated_prompt_marker(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(
        json.dumps(
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_a",
                    "name": "list_files",
                    "content": "<workspace_context>\nsecret\n</workspace_context>",
                }
            ]
        ),
        encoding="utf-8",
    )

    try:
        SessionStore.load(path)
    except ValueError as exc:
        assert "generated prompt" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_session_store_load_rejects_user_content_none(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps([{"role": "user", "content": None}]), encoding="utf-8")

    try:
        SessionStore.load(path)
    except ValueError as exc:
        assert "content" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_session_store_load_rejects_assistant_content_none_without_tool_calls(tmp_path):
    for records in [
        [{"role": "assistant", "content": None}],
        [{"role": "assistant", "content": None, "tool_calls": []}],
    ]:
        path = tmp_path / "session.json"
        path.write_text(json.dumps(records), encoding="utf-8")

        try:
            SessionStore.load(path)
        except ValueError as exc:
            assert "tool_calls" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError")


def test_session_store_load_rejects_malformed_assistant_tool_calls(tmp_path):
    malformed_tool_calls = [
        {"id": "call_a", "type": "file_search", "function": {"name": "list_files", "arguments": "{}"}},
        {"id": "call_a", "type": "function"},
        {"id": "call_a", "type": "function", "function": {"arguments": "{}"}},
        {"id": "call_a", "type": "function", "function": {"name": "", "arguments": "{}"}},
        {"id": "call_a", "type": "function", "function": {"name": "list_files"}},
        {"id": "call_a", "type": "function", "function": {"name": "list_files", "arguments": {}}},
    ]

    for tool_call_payload in malformed_tool_calls:
        path = tmp_path / "session.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call_payload],
                    }
                ]
            ),
            encoding="utf-8",
        )

        try:
            SessionStore.load(path)
        except ValueError as exc:
            assert "tool_calls" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError")


def test_session_store_load_accepts_assistant_tool_calls_and_tool_response(tmp_path):
    records = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("list_files", "{}", call_id="call_a")],
        },
        {
            "role": "tool",
            "tool_call_id": "call_a",
            "name": "list_files",
            "content": json.dumps({"ok": True}),
        },
    ]
    path = tmp_path / "session.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    assert SessionStore.load(path) == records


def test_session_store_latest_and_load_latest_use_newest_session_mtime(tmp_path):
    store = SessionStore(tmp_path)
    store.sessions_dir.mkdir(parents=True)
    older = store.sessions_dir / "20260513-120000-999999.json"
    newer = store.sessions_dir / "20260513-120000-000001.json"
    older.write_text(json.dumps([{"role": "user", "content": "older"}]), encoding="utf-8")
    newer.write_text(json.dumps([{"role": "user", "content": "newer"}]), encoding="utf-8")
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    assert store.latest() == newer
    assert store.load_latest() == [{"role": "user", "content": "newer"}]


def test_session_store_load_latest_returns_none_when_no_sessions(tmp_path):
    store = SessionStore(tmp_path)

    assert store.latest() is None
    assert store.load_latest() is None
