import sys

from coding_agent.policy import CommandPolicy
from coding_agent.sandbox import WorkspaceSandbox
from coding_agent.shell import ShellRunner
from coding_agent.tools import ToolRegistry, create_default_tools


def make_tools(tmp_path, allow=None):
    sandbox = WorkspaceSandbox(tmp_path)
    shell = ShellRunner(CommandPolicy(allow=allow or [], deny=[]), cwd=tmp_path)
    return create_default_tools(sandbox, shell)


def test_registry_returns_openai_function_schemas():
    registry = ToolRegistry()
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    registry.register("read_file", "Read a file", parameters, lambda arguments: {"ok": True})

    assert registry.schemas() == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": parameters,
            },
        }
    ]


def test_write_and_read_file_tools_use_utf8(tmp_path):
    tools = make_tools(tmp_path)

    write_result = tools.call("write_file", {"path": "notes/example.txt", "content": "hello\nworld"})
    read_result = tools.call("read_file", {"path": "notes/example.txt"})

    assert write_result == {"ok": True, "path": "notes/example.txt"}
    assert read_result == {"ok": True, "path": "notes/example.txt", "content": "hello\nworld"}


def test_list_files_returns_sorted_project_relative_posix_paths(tmp_path):
    tools = make_tools(tmp_path)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "pkg" / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "pkg" / "nested").mkdir()
    (tmp_path / "pkg" / "nested" / "c.txt").write_text("c", encoding="utf-8")

    result = tools.call("list_files", {"path": "pkg"})

    assert result == {"ok": True, "files": ["pkg/a.txt", "pkg/b.txt", "pkg/nested/c.txt"]}


def test_list_files_returns_error_for_missing_path(tmp_path):
    tools = make_tools(tmp_path)

    result = tools.call("list_files", {"path": "missing"})

    assert result["ok"] is False
    assert "missing" in result["error"].lower() or "not found" in result["error"].lower()
    assert "path" in result["error"].lower()


def test_search_text_finds_matches_and_skips_non_utf8_files(tmp_path):
    tools = make_tools(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.txt").write_text("alpha\nneedle here\n", encoding="utf-8")
    (tmp_path / "src" / "two.txt").write_text("another needle\n", encoding="utf-8")
    (tmp_path / "src" / "binary.dat").write_bytes(b"\xff\xfe\xfd")

    result = tools.call("search_text", {"query": "needle", "path": "src"})

    assert result == {
        "ok": True,
        "matches": [
            {"path": "src/one.txt", "line": 2, "text": "needle here"},
            {"path": "src/two.txt", "line": 1, "text": "another needle"},
        ],
    }


def test_unknown_tool_returns_error(tmp_path):
    tools = make_tools(tmp_path)

    result = tools.call("missing", {})

    assert result == {"ok": False, "error": "Unknown tool: missing"}


def test_sandbox_rejection_is_returned_as_tool_error(tmp_path):
    tools = make_tools(tmp_path)

    result = tools.call("read_file", {"path": "../outside.txt"})

    assert result["ok"] is False
    assert "outside workspace" in result["error"]


def test_apply_patch_is_explicitly_unsupported(tmp_path):
    tools = make_tools(tmp_path)

    result = tools.call("apply_patch", {"patch": "*** Begin Patch\n*** End Patch\n"})

    assert result["ok"] is False
    assert "unsupported" in result["error"].lower()


def test_run_shell_reports_rejected_command(tmp_path):
    tools = make_tools(tmp_path)

    result = tools.call("run_shell", {"command": f"{sys.executable} -c \"print('nope')\""})

    assert result["ok"] is False
    assert result["command"].startswith(sys.executable)
    assert result["exit_code"] is None
    assert result["stdout"] == ""
    assert "not in allow list" in result["stderr"]
    assert result["timed_out"] is False


def test_run_shell_reports_allowed_command(tmp_path):
    tools = make_tools(tmp_path, allow=[sys.executable])

    result = tools.call("run_shell", {"command": f"{sys.executable} -c \"print('ok')\""})

    assert result == {
        "ok": True,
        "command": f"{sys.executable} -c \"print('ok')\"",
        "exit_code": 0,
        "stdout": "ok\n",
        "stderr": "",
        "timed_out": False,
    }


def test_tool_exceptions_are_returned_as_errors():
    registry = ToolRegistry()
    registry.register("boom", "Raise", {"type": "object", "properties": {}}, lambda arguments: 1 / 0)

    result = registry.call("boom", {})

    assert result == {"ok": False, "error": "division by zero"}
