import subprocess
import sys

from coding_agent.policy import CommandPolicy
from coding_agent import shell
from coding_agent.shell import ShellRunner


def test_rejects_unlisted_command_without_executing(tmp_path):
    marker = tmp_path / "marker.txt"
    runner = ShellRunner(CommandPolicy(allow=[], deny=[]), cwd=tmp_path)

    result = runner.run(f"{sys.executable} -c \"from pathlib import Path; Path('marker.txt').write_text('ran')\"")

    assert result.allowed is False
    assert result.exit_code is None
    assert result.stdout == ""
    assert "not in allow list" in result.stderr
    assert not marker.exists()


def test_runs_allowed_command_in_cwd(tmp_path):
    runner = ShellRunner(CommandPolicy(allow=[sys.executable], deny=[]), cwd=tmp_path)

    result = runner.run(f"{sys.executable} -c \"from pathlib import Path; print(Path.cwd())\"")

    assert result.allowed is True
    assert result.exit_code == 0
    assert str(tmp_path) in result.stdout.strip()
    assert result.stderr == ""


def test_denied_command_is_not_executed(tmp_path):
    marker = tmp_path / "marker.txt"
    runner = ShellRunner(
        CommandPolicy(allow=[sys.executable], deny=[f"{sys.executable} -c"]),
        cwd=tmp_path,
    )

    result = runner.run(f"{sys.executable} -c \"from pathlib import Path; Path('marker.txt').write_text('ran')\"")

    assert result.allowed is False
    assert result.exit_code is None
    assert result.stdout == ""
    assert "denied" in result.stderr.lower()
    assert not marker.exists()


def test_bare_allowed_command_does_not_execute_workspace_batch_file(tmp_path):
    marker = tmp_path / "marker.txt"
    fake_pytest = tmp_path / "pytest.bat"
    fake_pytest.write_text("@echo off\necho ran > marker.txt\n", encoding="utf-8")
    runner = ShellRunner(CommandPolicy(allow=["pytest"], deny=[]), cwd=tmp_path)

    result = runner.run("pytest --version")

    assert not marker.exists()
    assert result.allowed is True


def test_rejects_chained_command_without_executing_second_segment(tmp_path):
    marker = tmp_path / "marker.txt"
    runner = ShellRunner(CommandPolicy(allow=["pytest"], deny=["del marker.txt"]), cwd=tmp_path)

    result = runner.run("pytest & del marker.txt")

    assert result.allowed is False
    assert result.exit_code is None
    assert "shell control operator" in result.stderr
    assert not marker.exists()


def test_rejects_single_quoted_chained_command_without_executing_second_segment(tmp_path):
    marker = tmp_path / "marker.txt"
    runner = ShellRunner(CommandPolicy(allow=["pytest"], deny=[]), cwd=tmp_path)

    result = runner.run("pytest '& echo ran > marker.txt'")

    assert result.allowed is False
    assert result.exit_code is None
    assert "shell control operator" in result.stderr
    assert not marker.exists()


def test_timeout_returns_timed_out_result(tmp_path):
    runner = ShellRunner(CommandPolicy(allow=[sys.executable], deny=[]), cwd=tmp_path, timeout_seconds=0.1)

    result = runner.run(f"{sys.executable} -c \"import time; print('started'); time.sleep(5)\"")

    assert result.allowed is True
    assert result.exit_code is None
    assert result.timed_out is True
    assert "timed out" in result.stderr.lower()


def test_timeout_decodes_captured_bytes(tmp_path, monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="example", timeout=1, output=b"out", stderr=b"err")

    monkeypatch.setattr(shell.subprocess, "run", raise_timeout)
    runner = ShellRunner(CommandPolicy(allow=[sys.executable], deny=[]), cwd=tmp_path, timeout_seconds=1)

    result = runner.run(f"{sys.executable} --version")

    assert result.stdout == "out"
    assert result.stderr.startswith("err\n")
