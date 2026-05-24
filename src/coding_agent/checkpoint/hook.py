from __future__ import annotations

from typing import Any

from coding_agent.checkpoint.patch_targets import parse_patch_targets
from coding_agent.checkpoint.store import CheckpointConflict, CheckpointStore
from coding_agent.orchestrator.lifecycle import AgentLifecycleHook, AgentTurnContext, ToolVeto


class CheckpointHook(AgentLifecycleHook):
    def __init__(self, store: CheckpointStore) -> None:
        self.store = store

    def pre_tool(self, ctx: AgentTurnContext, tool_name: str, args: dict[str, Any]) -> None:
        try:
            if tool_name == "write_file":
                self.store.verify_workspace_not_drifted()
                path = str(args.get("path", ""))
                if not path:
                    return
                self.store.verify_file_not_drifted(path)
                return
            if tool_name == "apply_patch":
                self.store.verify_workspace_not_drifted()
                for target in parse_patch_targets(str(args.get("patch", ""))):
                    if target.operation == "add":
                        self.store.verify_file_absent_for_add(target.path)
                    elif target.operation == "move_target":
                        self.store.verify_file_absent_for_add(target.path)
                    else:
                        self.store.verify_file_not_drifted(target.path)
        except CheckpointConflict as exc:
            raise ToolVeto(exc.result) from exc

    def after_tool(self, ctx: AgentTurnContext, tool_name: str, args: dict[str, Any], result: dict[str, Any]) -> None:
        if result.get("ok") is not True:
            return
        if tool_name == "read_file":
            path = result.get("path") or args.get("path")
            if isinstance(path, str) and path:
                self.store.record_file(path, source="read_file", run_id=ctx.run_id, session_id=ctx.session_id)
            return
        if tool_name == "write_file":
            path = result.get("path") or args.get("path")
            if isinstance(path, str) and path:
                self.store.record_file(path, source="write_file", run_id=ctx.run_id, session_id=ctx.session_id)
                self.store.refresh_workspace(run_id=ctx.run_id, session_id=ctx.session_id)
            return
        if tool_name == "apply_patch":
            changed = result.get("changed_files") or []
            if isinstance(changed, list):
                for path in changed:
                    if isinstance(path, str) and path:
                        self.store.record_file(path, source="apply_patch", run_id=ctx.run_id, session_id=ctx.session_id)
            for target in parse_patch_targets(str(args.get("patch", ""))):
                if target.operation in {"delete", "move_source"}:
                    self.store.record_deleted(target.path, source="apply_patch", run_id=ctx.run_id, session_id=ctx.session_id)
            self.store.refresh_workspace(run_id=ctx.run_id, session_id=ctx.session_id)
