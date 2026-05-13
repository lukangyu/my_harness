from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

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

    def __post_init__(self) -> None:
        object.__setattr__(self, "cwd", Path(self.cwd))

    def run(self, command: str) -> ShellResult:
        policy_result = self.policy.evaluate(command)
        if policy_result.decision is not CommandDecision.ALLOW:
            return ShellResult(
                command=command,
                allowed=False,
                exit_code=None,
                stdout="",
                stderr=policy_result.reason,
            )

        try:
            completed = subprocess.run(
                command,
                cwd=self.cwd,
                shell=True,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
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
