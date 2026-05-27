from pathlib import Path

from coding_agent.application import Application
from coding_agent.config import (
    AgentConfig,
    AppConfig,
    CommandConfig,
    ContextConfig,
    ModelConfig,
    WorkspaceConfig,
)
from coding_agent.orchestrator.agent_loop import AgentResult
from coding_agent.run_result import RunTaskResult


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        project_root=tmp_path,
        model=ModelConfig(
            base_url="https://example.test/v1",
            api_key_env="TEST_KEY",
            api_key="test-key",
            model="test-model",
        ),
        agent=AgentConfig(max_steps=3, stream=False),
        workspace=WorkspaceConfig(root=tmp_path),
        commands=CommandConfig(allow=[], deny=[]),
        context=ContextConfig(
            include_project_docs=False,
            include_file_tree=False,
            include_git_status=False,
            include_recent_commits=False,
        ),
    )


def test_application_imports_and_run_task_constructs_runtime(tmp_path, monkeypatch):
    progress_events = []
    shell_runners = []

    class FakeAgentLoop:
        def __init__(self, **kwargs):
            self.on_progress = kwargs["on_progress"]
            self.tools = kwargs["tools"]

        def run(self, task, *, prior_messages, mode):
            tool_names = [schema["function"]["name"] for schema in self.tools.schemas()]
            self.on_progress({"type": "model_attempt", "attempts": 1, "step": 1})
            progress_events.append(
                {"task": task, "prior_messages": prior_messages, "mode": mode, "tool_names": tool_names}
            )
            return AgentResult(
                final_answer="done",
                messages=[],
                conversation_messages=[{"role": "assistant", "content": "done"}],
                attempts=1,
                tool_steps=0,
            )

    class FakeShellRunner:
        def __init__(self, *, policy, cwd, approval_callback=None):
            self.policy = policy
            self.cwd = cwd
            self.approval_callback = approval_callback
            shell_runners.append(self)

    monkeypatch.setattr("coding_agent.application.AgentLoop", FakeAgentLoop)
    monkeypatch.setattr("coding_agent.application.ShellRunner", FakeShellRunner)
    approval = lambda command, reason: True

    result = Application(make_config(tmp_path), command_approval=approval).run_task("inspect", [], "run")

    assert isinstance(result, RunTaskResult)
    assert result.result.final_answer == "done"
    assert result.session_path.exists()
    assert result.run_dir is not None
    assert (result.run_dir / "audit" / "task_state.json").exists()
    assert progress_events == [
        {
            "task": "inspect",
            "prior_messages": [],
            "mode": "run",
            "tool_names": [
                "list_files",
                "read_file",
                "write_file",
                "search_text",
                "apply_patch",
                "run_shell",
                "session_search",
                "start_subagent",
                "wait_subagent",
                "cancel_subagent",
            ],
        }
    ]
    assert shell_runners[0].approval_callback is approval


def test_cli_runtask_result_uses_shared_type():
    import coding_agent.cli as cli_module

    assert cli_module.RunTaskResult is RunTaskResult
