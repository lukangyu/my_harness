from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any


MEMORY_TYPES = {"personal", "procedural", "knowledge"}
MIN_CONFIDENCE = 0.2


class LongTermMemoryStore:
    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.raw_dir = memory_dir / "raw"

    def append_memories(
        self,
        memories: list[dict[str, Any]],
        *,
        source: str,
        evidence: list[str],
    ) -> list[dict[str, Any]]:
        records = [
            record
            for memory in memories
            if (record := _normalize_memory(memory, source=source, evidence=evidence)) is not None
        ]
        if not records:
            return []
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        path = self.raw_dir / f"{time.strftime('%Y-%m-%d', time.localtime())}.jsonl"
        with path.open("a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return records

    def search_memories(self, query: str, *, max_results: int = 3) -> list[dict[str, Any]]:
        terms = _terms(query)
        if not terms or not self.raw_dir.exists():
            return []
        scored: list[tuple[int, dict[str, Any]]] = []
        for path in sorted(self.raw_dir.glob("*.jsonl")):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                haystack = " ".join(
                    str(record.get(key) or "")
                    for key in ("type", "content", "reason", "source")
                ).lower()
                score = sum(1 for term in terms if term in haystack)
                if score <= 0:
                    continue
                scored.append(
                    (
                        score,
                        {
                            "type": record.get("type"),
                            "content": record.get("content"),
                            "source": _relative_source(path, self.memory_dir),
                            "confidence": record.get("confidence"),
                        },
                    )
                )
        scored.sort(key=lambda item: (-item[0], str(item[1].get("source") or "")))
        return [record for _, record in scored[:max_results]]


def _normalize_memory(
    memory: dict[str, Any],
    *,
    source: str,
    evidence: list[str],
) -> dict[str, Any] | None:
    memory_type = memory.get("type")
    content = memory.get("content")
    reason = memory.get("reason")
    confidence = memory.get("confidence")
    if memory_type not in MEMORY_TYPES:
        return None
    if not isinstance(content, str) or not content.strip():
        return None
    if not isinstance(reason, str) or not reason.strip():
        return None
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return None
    confidence_float = max(0.0, min(1.0, float(confidence)))
    if confidence_float < MIN_CONFIDENCE:
        return None
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    return {
        "id": f"mem_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{uuid.uuid4().hex[:8]}",
        "type": memory_type,
        "content": content.strip(),
        "reason": reason.strip(),
        "confidence": confidence_float,
        "source": source,
        "evidence": [item for item in evidence if isinstance(item, str) and item],
        "created_at": now,
    }


def _terms(query: str) -> list[str]:
    if not isinstance(query, str):
        return []
    normalized = query.lower()
    terms = [part for part in re.split(r"\W+", normalized) if len(part) >= 2]
    if not terms and normalized.strip():
        terms = [normalized.strip()[:100]]
    return terms[:12]


def _relative_source(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
