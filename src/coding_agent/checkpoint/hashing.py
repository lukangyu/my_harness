from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from coding_agent.execution.sandbox import WorkspaceSandbox


@dataclass(frozen=True)
class FileState:
    path: str
    exists: bool
    content_hash: str | None
    size: int | None
    mtime_ns: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def file_state(path: str | Path, sandbox: WorkspaceSandbox) -> FileState:
    resolved = sandbox.resolve(path)
    relative = sandbox.relative_path(resolved)
    if not resolved.exists():
        return FileState(path=relative, exists=False, content_hash=None, size=None, mtime_ns=None)
    if not resolved.is_file():
        raise IsADirectoryError(f"path is not a file: {relative}")
    stat = resolved.stat()
    return FileState(
        path=relative,
        exists=True,
        content_hash=_sha256_file(resolved),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
