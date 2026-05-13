import json
from pathlib import Path

import httpx

from coding_agent.agent import AgentLoop
from coding_agent.llm import LLMError, OpenAICompatibleClient
from coding_agent.policy import CommandPolicy
from coding_agent.sandbox import WorkspaceSandbox
from coding_agent.session import SessionStore
from coding_agent.shell import ShellRunner
from coding_agent.tools import create_default_tools


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
                }
            },
            {"message": {"role": "assistant", "content": "done"}},
        ]
    )
    agent = AgentLoop(client, make_tools(tmp_path), max_steps=3)

    result = agent.run("write a note")

    assert result.final_answer == "done"
    assert result.reached_max_steps is False
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello"
    assert client.calls[0]["messages"][0]["role"] == "system"
    assert client.calls[0]["messages"][1] == {"role": "user", "content": "write a note"}
    assert client.calls[1]["messages"][-1]["role"] == "tool"
    assert client.calls[1]["messages"][-1]["tool_call_id"] == "call_1"
    assert client.calls[1]["messages"][-1]["name"] == "write_file"
    assert json.loads(client.calls[1]["messages"][-1]["content"]) == {
        "ok": True,
        "path": "notes.txt",
    }


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
    agent = AgentLoop(client, make_tools(tmp_path), max_steps=3)

    result = agent.run("write a note")

    tool_message = client.calls[1]["messages"][-1]
    tool_result = json.loads(tool_message["content"])
    assert result.final_answer == "fixed"
    assert tool_result["ok"] is False
    assert "invalid json" in tool_result["error"].lower()


def test_agent_stops_after_max_steps(tmp_path):
    client = FakeClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("list_files", "{}")],
                }
            }
        ]
    )
    agent = AgentLoop(client, make_tools(tmp_path), max_steps=1)

    result = agent.run("inspect files")

    assert result.reached_max_steps is True
    assert "max steps" in result.final_answer.lower()


def test_agent_continues_prior_messages_without_system_prompt(tmp_path):
    client = FakeClient([{"message": {"role": "assistant", "content": "answer"}}])
    agent = AgentLoop(client, make_tools(tmp_path), max_steps=1)
    prior_messages = [{"role": "user", "content": "earlier"}]

    result = agent.run("new task", prior_messages=prior_messages)

    assert result.final_answer == "answer"
    assert client.calls[0]["messages"] == [
        {"role": "user", "content": "earlier"},
        {"role": "user", "content": "new task"},
    ]


def test_llm_client_posts_openai_compatible_chat_request(monkeypatch):
    requests = []

    def fake_post(url, *, headers, json, timeout):
        requests.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"role": "assistant", "content": "hello"}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OpenAICompatibleClient("https://example.test/v1/", "secret", "model-a", timeout=12.5)

    result = client.chat([{"role": "user", "content": "hi"}], [{"type": "function"}])

    assert result == {"message": {"role": "assistant", "content": "hello"}}
    assert requests == [
        {
            "url": "https://example.test/v1/chat/completions",
            "headers": {"Authorization": "Bearer secret"},
            "json": {
                "model": "model-a",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function"}],
                "tool_choice": "auto",
            },
            "timeout": 12.5,
        }
    ]


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
