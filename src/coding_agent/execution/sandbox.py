from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkspaceSandbox:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.root / candidate

        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise SandboxError(f"Path is outside workspace: {path}") from exc

        return resolved

    def relative_path(self, path: str | Path) -> str:
        return self.resolve(path).relative_to(self.root).as_posix()
