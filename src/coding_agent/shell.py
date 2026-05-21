from __future__ import annotations

import subprocess
import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from coding_agent.policy import CommandDecision, CommandPolicy


@dataclass(frozen=True)
class ShellResult:
    command: str
    allowed: bool
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False


@dataclass(frozen=True)
class ShellRunner:
    policy: CommandPolicy
    cwd: Path
    timeout_seconds: float = 120
    approval_callback: Callable[[str, str], bool] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cwd", Path(self.cwd))

    def run(self, command: str) -> ShellResult:
        policy_result = self.policy.evaluate(command)
        if policy_result.decision is not CommandDecision.ALLOW:
            if not _can_request_approval(policy_result.reason):
                return ShellResult(
                    command=command,
                    allowed=False,
                    exit_code=None,
                    stdout="",
                    stderr=policy_result.reason,
                )
            if self.approval_callback is None:
                return ShellResult(
                    command=command,
                    allowed=False,
                    exit_code=None,
                    stdout="",
                    stderr=policy_result.reason,
                )
            if not self.approval_callback(command, policy_result.reason):
                return ShellResult(
                    command=command,
                    allowed=False,
                    exit_code=None,
                    stdout="",
                    stderr="Command rejected by human approval",
                )

        try:
            argv = _command_to_argv(command, self.cwd)
            completed = subprocess.run(
                argv,
                cwd=self.cwd,
                shell=False,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
            )
        except ValueError as exc:
            return ShellResult(
                command=command,
                allowed=False,
                exit_code=None,
                stdout="",
                stderr=str(exc),
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _to_text(exc.stdout)
            stderr = _to_text(exc.stderr)
            timeout_message = f"Command timed out after {self.timeout_seconds} seconds"
            if stderr:
                stderr = f"{stderr}\n{timeout_message}"
            else:
                stderr = timeout_message
            return ShellResult(
                command=command,
                allowed=True,
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
            )

        return ShellResult(
            command=command,
            allowed=True,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def _to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _command_to_argv(command: str, cwd: Path) -> list[str]:
    try:
        argv = [_strip_outer_quotes(arg) for arg in shlex.split(command, posix=os.name != "nt")]
    except ValueError as exc:
        raise ValueError(f"Invalid shell command syntax: {exc}") from exc
    if not argv:
        raise ValueError("Command is empty")

    executable = argv[0]
    resolved = shutil.which(executable, path=os.environ.get("PATH"))
    if resolved is None:
        raise ValueError(f"Executable not found on PATH: {executable}")

    resolved_path = Path(resolved).resolve()
    cwd_path = cwd.resolve()
    if resolved_path == cwd_path or cwd_path in resolved_path.parents:
        raise ValueError(f"Refusing to execute workspace-local command: {executable}")

    return [str(resolved_path), *argv[1:]]


def _strip_outer_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _can_request_approval(reason: str) -> bool:
    return reason == "Command not in allow list"
