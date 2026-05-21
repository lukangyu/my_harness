# Telemetry Logs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add local JSONL event logs and trace timing logs so users can understand the full coding-agent execution flow in Chinese.

**Architecture:** Add a small `TelemetryLogger` module that appends `events.jsonl` and `trace.jsonl` under `.coding-agent/logs`. Pass it from CLI into AgentLoop, LLM client, tools, and SessionStore without changing model prompts or tool contracts.

**Tech Stack:** Python stdlib (`json`, `time`, `uuid`, `contextlib`, `pathlib`) plus existing project classes.

---

### Task 1: Telemetry Core

**Files:**
- Create: `src/coding_agent/telemetry.py`
- Test: `tests/test_telemetry.py`

- [ ] Create `TelemetryLogger` with `event()`, `span()`, and `workspace_snapshot()` methods.
- [ ] Write JSONL records in UTF-8 with `ensure_ascii=False`.
- [ ] Include Chinese `message_zh`, function name, phase, metadata, and duration.

### Task 2: Runtime Integration

**Files:**
- Modify: `src/coding_agent/cli.py`
- Modify: `src/coding_agent/agent.py`
- Modify: `src/coding_agent/llm.py`
- Modify: `src/coding_agent/tools.py`
- Modify: `src/coding_agent/session.py`

- [ ] Create telemetry after config is loaded.
- [ ] Log CLI task start/end, workspace snapshots, agent steps, LLM calls, tool calls, and session saves.
- [ ] Keep all telemetry optional so existing tests and direct class usage still work.

### Task 3: Verification

**Files:**
- Test: `tests/test_telemetry.py`
- Test: existing tests

- [ ] Verify event and trace JSONL files are created.
- [ ] Verify workspace tree snapshot includes files and directories.
- [ ] Verify sensitive data such as API keys is not written.
