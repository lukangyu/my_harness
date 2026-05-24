from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PatchOperation = Literal["add", "update", "delete", "move_source", "move_target"]


@dataclass(frozen=True)
class PatchTarget:
    path: str
    operation: PatchOperation


def parse_patch_targets(patch: str) -> list[PatchTarget]:
    lines = patch.splitlines()
    if not lines or lines[0] != "*** Begin Patch":
        raise ValueError("patch must start with *** Begin Patch")
    if lines[-1] != "*** End Patch":
        raise ValueError("patch must end with *** End Patch")

    targets: list[PatchTarget] = []
    index = 1
    while index < len(lines) - 1:
        line = lines[index]
        if line.startswith("*** Add File: "):
            targets.append(PatchTarget(line.removeprefix("*** Add File: ").strip(), "add"))
            index += 1
            continue
        if line.startswith("*** Delete File: "):
            targets.append(PatchTarget(line.removeprefix("*** Delete File: ").strip(), "delete"))
            index += 1
            continue
        if line.startswith("*** Update File: "):
            source = line.removeprefix("*** Update File: ").strip()
            if index + 1 < len(lines) and lines[index + 1].startswith("*** Move to: "):
                targets.append(PatchTarget(source, "move_source"))
                targets.append(PatchTarget(lines[index + 1].removeprefix("*** Move to: ").strip(), "move_target"))
                index += 2
            else:
                targets.append(PatchTarget(source, "update"))
                index += 1
            continue
        index += 1
    return targets
