import sys
import inspect
from pathlib import Path
from types import SimpleNamespace

from coding_agent.execution.policy import CommandPolicy
from coding_agent.execution.sandbox import WorkspaceSandbox
from coding_agent.execution.shell import ShellRunner
from coding_agent.execution.tools import ToolRegistry, create_default_tools


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

    assert write_result["ok"] is True
    assert write_result["path"] == "notes/example.txt"
    assert read_result["ok"] is True
    assert read_result["path"] == "notes/example.txt"
    assert read_result["content"] == "hello\nworld"
    assert read_result["metadata"]["total_lines"] == 2


def test_list_files_returns_sorted_project_relative_posix_paths(tmp_path):
    tools = make_tools(tmp_path)
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "pkg" / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "pkg" / "nested").mkdir()
    (tmp_path / "pkg" / "nested" / "c.txt").write_text("c", encoding="utf-8")

    result = tools.call("list_files", {"path": "pkg"})

    assert result["ok"] is True
    assert result["files"] == ["pkg/a.txt", "pkg/b.txt", "pkg/nested/c.txt"]
    assert result["metadata"]["truncated"] is False


def test_list_files_applies_limit_depth_and_default_ignores(tmp_path):
    tools = make_tools(tmp_path)
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("ignored", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "pkg" / "c.txt").write_text("c", encoding="utf-8")
    (tmp_path / "pkg" / "nested").mkdir()
    (tmp_path / "pkg" / "nested" / "b.txt").write_text("b", encoding="utf-8")

    result = tools.call("list_files", {"path": ".", "max_depth": 2, "max_entries": 1})

    assert result["ok"] is True
    assert result["files"] == ["pkg/a.txt"]
    assert result["metadata"]["truncated"] is True
    assert ".git/config" not in result["files"]


def test_read_file_supports_inclusive_line_range(tmp_path):
    tools = make_tools(tmp_path)
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")

    result = tools.call(
        "read_file",
        {"path": "notes.txt", "start_line": 2, "end_line": 3},
    )

    assert result["ok"] is True
    assert result["content"] == "two\nthree\n"
    assert result["metadata"] == {
        "start_line": 2,
        "end_line": 3,
        "total_lines": 4,
        "returned_chars": 10,
        "truncated": False,
        "next_start_line": None,
        "notice": None,
    }


def test_read_file_truncates_large_output_with_resume_hint(tmp_path):
    tools = make_tools(tmp_path)
    (tmp_path / "large.txt").write_text("line1\nline2\nline3\n", encoding="utf-8")

    result = tools.call("read_file", {"path": "large.txt", "max_chars": 8})

    assert result["ok"] is True
    assert result["content"] == "line1\nli"
    assert result["metadata"] == {
        "start_line": 1,
        "end_line": 2,
        "total_lines": 3,
        "returned_chars": 8,
        "truncated": True,
        "next_start_line": 2,
        "notice": "Output truncated at 8 chars. Continue with start_line=2.",
    }


def test_read_file_schema_uses_end_line_and_50kb_default():
    tools = make_tools(Path.cwd())
    read_schema = next(
        schema for schema in tools.schemas() if schema["function"]["name"] == "read_file"
    )

    properties = read_schema["function"]["parameters"]["properties"]
    assert "end_line" in properties
    assert "line_count" not in properties
    assert properties["max_chars"]["default"] == 50_000


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

    assert result["ok"] is True
    assert result["matches"] == [
        {"path": "src/one.txt", "line": 2, "text": "needle here"},
        {"path": "src/two.txt", "line": 1, "text": "another needle"},
    ]


def test_tool_registry_and_default_tools_do_not_accept_memory_or_telemetry_dependencies():
    registry_signature = inspect.signature(ToolRegistry)
    default_tools_signature = inspect.signature(create_default_tools)

    assert "memory_store" not in registry_signature.parameters
    assert "telemetry" not in registry_signature.parameters
    assert "memory_store" not in default_tools_signature.parameters
    assert "telemetry" not in default_tools_signature.parameters


def test_search_text_supports_regex_case_and_glob_limit(tmp_path):
    tools = make_tools(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.py").write_text("Alpha\nneedle_123\n", encoding="utf-8")
    (tmp_path / "src" / "three.py").write_text("needle_789\n", encoding="utf-8")
    (tmp_path / "src" / "two.txt").write_text("needle_456\n", encoding="utf-8")

    result = tools.call(
        "search_text",
        {
            "query": "NEEDLE_[0-9]+",
            "path": "src",
            "regex": True,
            "case_sensitive": False,
            "glob": "*.py",
            "max_matches": 1,
        },
    )

    assert result["ok"] is True
    assert result["matches"] == [{"path": "src/one.py", "line": 2, "text": "needle_123"}]
    assert result["metadata"]["truncated"] is True


def test_search_text_uses_rg_when_available(tmp_path, monkeypatch):
    tools = make_tools(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.txt").write_text("needle\n", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="one.txt:1:needle\n")

    monkeypatch.setattr("coding_agent.execution.tools.shutil.which", lambda name: "rg.exe" if name == "rg" else None)
    monkeypatch.setattr("coding_agent.execution.tools.subprocess.run", fake_run)

    result = tools.call("search_text", {"query": "needle", "path": "src", "regex": False})

    assert result["ok"] is True
    assert result["matches"] == [{"path": "src/one.txt", "line": 1, "text": "needle"}]
    assert result["metadata"]["engine"] == "rg"
    assert "--fixed-strings" in calls[0][0]
    assert calls[0][1]["cwd"] == tmp_path / "src"


def test_search_text_falls_back_when_rg_is_unavailable(tmp_path, monkeypatch):
    tools = make_tools(tmp_path)
    (tmp_path / "notes.txt").write_text("needle\n", encoding="utf-8")
    monkeypatch.setattr("coding_agent.execution.tools.shutil.which", lambda name: None)

    result = tools.call("search_text", {"query": "needle", "path": "."})

    assert result["ok"] is True
    assert result["matches"] == [{"path": "notes.txt", "line": 1, "text": "needle"}]
    assert result["metadata"]["engine"] == "python"


def test_unknown_tool_returns_error(tmp_path):
    tools = make_tools(tmp_path)

    result = tools.call("missing", {})

    assert result == {"ok": False, "error": "Unknown tool: missing"}


def test_sandbox_rejection_is_returned_as_tool_error(tmp_path):
    tools = make_tools(tmp_path)

    result = tools.call("read_file", {"path": "../outside.txt"})

    assert result["ok"] is False
    assert "outside workspace" in result["error"]


def test_apply_patch_updates_adds_and_deletes_files(tmp_path):
    tools = make_tools(tmp_path)
    (tmp_path / "old.txt").write_text("remove me\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    result = tools.call(
        "apply_patch",
        {
            "patch": "\n".join(
                [
                    "*** Begin Patch",
                    "*** Update File: notes.txt",
                    "@@",
                    " one",
                    "-two",
                    "+TWO",
                    " three",
                    "*** Add File: nested/new.txt",
                    "+created",
                    "*** Delete File: old.txt",
                    "*** End Patch",
                ]
            )
        },
    )

    assert result["ok"] is True
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "one\nTWO\nthree\n"
    assert (tmp_path / "nested" / "new.txt").read_text(encoding="utf-8") == "created\n"
    assert not (tmp_path / "old.txt").exists()
    assert result["changed_files"] == ["notes.txt", "nested/new.txt", "old.txt"]


def test_apply_patch_rejects_paths_outside_workspace(tmp_path):
    tools = make_tools(tmp_path)

    result = tools.call(
        "apply_patch",
        {"patch": "*** Begin Patch\n*** Add File: ../outside.txt\n+nope\n*** End Patch"},
    )

    assert result["ok"] is False
    assert "outside workspace" in result["error"]


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
        "metadata": {"elapsed_ms": result["metadata"]["elapsed_ms"]},
    }
    assert result["metadata"]["elapsed_ms"] >= 0


def test_tool_exceptions_are_returned_as_errors():
    registry = ToolRegistry()
    registry.register("boom", "Raise", {"type": "object", "properties": {}}, lambda arguments: 1 / 0)

    result = registry.call("boom", {})

    assert result == {"ok": False, "error": "division by zero"}
