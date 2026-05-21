from __future__ import annotations

from pathlib import Path
from typing import Callable

from coding_agent.agent import AgentLoop
from coding_agent.config import AppConfig
from coding_agent.context import WorkspaceContextOptions
from coding_agent.llm import OpenAICompatibleClient
from coding_agent.memory import MemoryStore
from coding_agent.policy import CommandPolicy
from coding_agent.run_coordinator import RunCoordinator
from coding_agent.run_result import RunTaskResult
from coding_agent.runtime_events import RuntimeEvent
from coding_agent.run_store import RunStore
from coding_agent.sandbox import WorkspaceSandbox
from coding_agent.session import SessionStore
from coding_agent.shell import ShellRunner
from coding_agent.telemetry import TelemetryLogger
from coding_agent.tools import create_default_tools


ToolPrinter = Callable[[dict], None]
ReasoningPrinter = Callable[[str], None]
RuntimeEventPrinter = Callable[[RuntimeEvent], None]
CommandApproval = Callable[[str, str], bool]


class Application:
    def __init__(
        self,
        config: AppConfig,
        *,
        on_reasoning_delta: ReasoningPrinter | None = None,
        on_tool_call: ToolPrinter | None = None,
        on_runtime_event: RuntimeEventPrinter | None = None,
        command_approval: CommandApproval | None = None,
    ) -> None:
        self.config = config
        self.on_reasoning_delta = on_reasoning_delta
        self.on_tool_call = on_tool_call
        self.on_runtime_event = on_runtime_event
        self.command_approval = command_approval

    def run_task(
        self,
        task: str,
        prior_messages: list[dict] | None,
        mode: str,
    ) -> RunTaskResult:
        sandbox = WorkspaceSandbox(self.config.workspace.root)
        run_store = RunStore(self.config.project_root)
        run_artifact, task_state = run_store.start_run(
            task=task,
            mode=mode,
            workspace_root=sandbox.root,
        )
        telemetry = TelemetryLogger(
            run_artifact.run_dir,
            workspace_root=sandbox.root,
            run_id=run_artifact.run_id,
        )
        memory_store = MemoryStore(
            self.config.project_root,
            dialog_dir=run_artifact.dialog_dir,
            tool_result_dir=run_artifact.tool_result_dir,
        )
        policy = CommandPolicy(allow=self.config.commands.allow, deny=self.config.commands.deny)
        shell = ShellRunner(policy=policy, cwd=sandbox.root, approval_callback=self.command_approval)
        tools = create_default_tools(sandbox, shell, telemetry=telemetry, memory_store=memory_store)
        client = OpenAICompatibleClient(
            base_url=self.config.model.base_url,
            api_key=self.config.model.api_key,
            model=self.config.model.model,
            stream=self.config.agent.stream,
            on_reasoning_delta=self.on_reasoning_delta,
            debug_dir=run_artifact.debug_dir,
            telemetry=telemetry,
        )
        coordinator = RunCoordinator(
            run_store=run_store,
            run_artifact=run_artifact,
            task_state=task_state,
            telemetry=telemetry,
            memory_store=memory_store,
            session_store=SessionStore(self.config.project_root, telemetry=telemetry),
        )
        agent = AgentLoop(
            client=client,
            tools=tools,
            max_steps=self.config.agent.max_steps,
            cwd=sandbox.root,
            context_options=_context_options(self.config),
            recent_message_tokens=self.config.context.recent_message_tokens,
            on_tool_call=self.on_tool_call,
            telemetry=telemetry,
            memory_store=memory_store,
            on_progress=coordinator.record_progress,
            on_runtime_event=self.on_runtime_event,
        )
        return coordinator.run(
            agent=agent,
            task=task,
            prior_messages=prior_messages,
            mode=mode,
            show_cache_stats=self.config.context.show_cache_stats,
            workspace_root=sandbox.root,
        )


def _context_options(config: AppConfig) -> WorkspaceContextOptions:
    return WorkspaceContextOptions(
        doc_max_chars=config.context.doc_max_chars,
        tree_max_entries=config.context.tree_max_entries,
        include_project_docs=config.context.include_project_docs,
        include_file_tree=config.context.include_file_tree,
        include_git_status=config.context.include_git_status,
        include_recent_commits=config.context.include_recent_commits,
        max_input_tokens=config.context.max_input_tokens,
        compact_threshold_ratio=config.context.compact_threshold_ratio,
        protected_recent_turns=config.context.protected_recent_turns,
        protected_tool_results=config.context.protected_tool_results,
        handoff_max_chars=config.context.handoff_max_chars,
        scratchpad_max_chars=config.context.scratchpad_max_chars,
        file_summaries_max_count=config.context.file_summaries_max_count,
        file_summaries_max_chars=config.context.file_summaries_max_chars,
    )
