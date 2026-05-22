from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coding_agent.orchestrator.agent_loop import AgentResult


@dataclass(frozen=True)
class RunTaskResult:
    result: AgentResult
    session_path: Path
    show_cache_stats: bool
    run_id: str | None = None
    run_dir: Path | None = None
