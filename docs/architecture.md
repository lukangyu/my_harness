# coding-agent 架构说明

`coding-agent` 的运行时按职责拆成五个域：

- **Orchestrator**：管理 run 生命周期和 LLM-Tool 循环。
- **Context**：管理结构化上下文、压缩和最终 messages 编译。
- **Execution**：执行工具调用，并用 sandbox/policy 限制副作用。
- **Checkpoint**：保存 conversation 级工作区乐观锁，防止陈旧读取后的脏写。
- **Memory**：保存 session 级 handoff 与 scratchpad。
- **Telemetry**：保存 run 工件、events、trace 和 debug payload。

设计原则：工具层只执行动作，Context 只保存数据，Checkpoint 只做写前安全断言，Memory 只存当前 session 的必要状态，Run artifacts 保存可审计原文。

## 存储模型

```text
.coding-agent/
  conversations/
    <conversation_id>/
      conversation.json
      checkpoints/
        checkpoint.json
      memory/
        raw/
          YYYY-MM-DD.jsonl
      sessions/
        <session_id>/
          session.json
          memory/
            handoff.md
            scratchpad.json
          runs/
            <run_id>/
              agent_context/
                dialog/
                  *.jsonl
                tool_result/
                  *.txt
              audit/
                task_state.json
                report.json
                events.jsonl
                debug/
```

语义：

- `conversation`：长期任务线。
- `checkpoint`：conversation 级工作区安全状态，跨 session rotation 保留。
- `conversation/memory`：跨 session 的长期记忆候选，由压缩时提取。
- `session`：上下文窗口 epoch。
- `run`：一次 AgentLoop 执行。
- `agent_context`：模型可通过路径回溯的归档上下文；`audit`：人类审计和实验统计工件，不由 `session_search` 主动暴露。

不保留派生缓存：

- `file_summaries.json`
- `tool_index.jsonl`
- `status.json`
- `current_session`
- `compact_summary.md`

旧 `.coding-agent/sessions/*.json` 不再支持 resume。

## 模块职责

### `application.py`

运行时装配中心：

- 创建或恢复 `ConversationStore` / `SessionRef`
- 将 `RunStore` 挂到 `session/runs`
- 将 `MemoryStore` 挂到 `session/memory`
- 将 `CheckpointStore` 挂到 `conversation/checkpoints`
- 注册默认工具和 `session_search`
- 通过 `AgentLifecycleRegistry` 按 phase/order 注册 `CheckpointHook`、`ContextCompactionHook`、`MemoryProjectionHook` 和 `ToolResultOffloadHook`
- 创建 `AgentLoop` 与 `RunCoordinator`

### `session.py`

提供 conversation/session 目录模型：

- `ConversationStore`
- `ConversationRef`
- `SessionRef`
- `SessionRuntime`

`ConversationStore.compact_session()` 负责 session epoch rotation：

1. 创建新 session 目录。
2. 写新 `session.json`。
3. 写新 `memory/handoff.md`。
4. 复制旧 `memory/scratchpad.json`。
5. 标记旧 session 为 `compacted`。
6. 最后原子写 `conversation.json.active_session_id`。

`conversation.json` 必须最后切换，避免中断后 active session 指向半成品目录。

### `orchestrator/agent_loop.py`

纯 LLM-Tool 状态机：

1. 构建 `Context`
2. 触发 `pre_llm`
3. `PromptBuilder` 生成 messages
4. 调 LLM
5. 执行工具调用
6. 追加 tool messages
7. 返回最终答案或达到 max steps

### `orchestrator/lifecycle.py`

生命周期 Hook 容器：

- `AgentLifecycleRegistry` 按 phase 注册 hook。
- `HookRegistration.order` 只在当前 phase 内生效。
- 相同 order 保留注册顺序。
- `after_tool` 是流水线阶段，hook 可以返回新的 result 交给后续 hook 和最终 tool message。

当前 phase：

```text
on_turn_start
pre_llm
after_llm
pre_tool
after_tool
on_turn_end
```

### `orchestrator/coordinator.py`

管理单次 run 的外层生命周期：

- 写 `task_state.json`
- 写 `report.json`
- 保存 active session messages
- 记录 workspace 快照
- 处理中断和异常状态

### `context/context.py`

`Context` 是结构化状态实体。它保存：

- workspace frame
- scratchpad frame
- handoff frame
- active history frames
- current task frame

`Context` 同时维护：

- `_raw_frames`：完整原始历史。
- `_active_frames`：送入模型的当前历史。

压缩只替换 active frames，不破坏 raw frames。

### `context/compressor.py`

`ContextCompressor` 在超出阈值时：

1. 切分旧历史和 protected recent messages。
2. 归档旧消息到 `runs/<run_id>/agent_context/dialog/*.jsonl`。
3. 用压缩模型生成 JSON：`handoff` 和长期 `memories`。
4. 写当前 memory handoff。
5. 将长期 memories 追加到 `conversation/memory/raw/YYYY-MM-DD.jsonl`。
6. 将 compact summary 作为 assistant 消息插入 active history。
7. 返回 `CompressionResult` 给 hook 做 session rotation。

如果模型没有返回合法 JSON，压缩器会把原始文本当作 handoff，`memories=[]`，避免记忆提取失败阻塞上下文压缩。

### `context/prompt_builder.py`

唯一最终 messages 工厂。输出顺序：

```text
system: 固定规训和工具边界
user: workspace / scratchpad / handoff
assistant/user/tool: active history
user: current task
```

工具 schema 不写进 prompt，仍通过 API `tools` 数组传递。

### `execution/tools.py`

工具注册和路由。默认工具：

- `list_files`
- `read_file`
- `write_file`
- `search_text`
- `apply_patch`
- `run_shell`
- `session_search`

工具函数不写 memory，不写 telemetry。

`session_search` 默认搜当前 conversation 下所有 sessions；`scope=current_session` 时只搜当前 active session。

### `execution/executor.py`

执行模型 tool call：

- 解析 JSON arguments
- 触发 `pre_tool`，允许 checkpoint veto 写操作
- 调用 `ToolRegistry`
- 触发 `after_tool` 流水线 hooks
- 生成 OpenAI tool message

`ToolExecutor` 不直接写 memory 或 telemetry；长输出 offload 由 `ToolResultOffloadHook` 负责。

### `hooks/tool_result_offload_hook.py`

`after_tool` 阶段靠后执行，把超长 tool result 转存到：

```text
session/runs/<run_id>/agent_context/tool_result/*.txt
```

最终返回给模型的 tool message 只保留 preview、路径和读取指引。

### `checkpoint/store.py`

`CheckpointStore` 管理 conversation 级安全状态：

```text
conversation/checkpoints/checkpoint.json
```

它保存两类信息：

- workspace identity：workspace root、git branch、HEAD、status、fingerprint。
- file records：按相对路径 upsert 的 last-seen 记录，包含 `exists`、sha256、size、mtime、来源 run/session。

删除文件不会移除记录，而是写 tombstone：

```json
{"path": "src/old.py", "exists": false, "content_hash": null}
```

Checkpoint 不进 prompt。它不是记忆系统，只在写工具执行前做系统级安全断言。

### `checkpoint/hook.py`

`CheckpointHook` 运行在 tool lifecycle：

- `after_tool(read_file)`：记录模型最后一次看到的文件状态。
- `pre_tool(write_file)`：校验 workspace 没漂移，目标文件未被外部修改。
- `pre_tool(apply_patch)`：解析 patch 目标，校验 update/delete/move source 已读且未漂移，add/move target 不存在。
- `after_tool(write_file/apply_patch)`：刷新相关 file records 和 workspace checkpoint。

发生冲突时不执行真实写入，而是返回结构化 tool error：

```json
{
  "ok": false,
  "code": "file_drift_detected",
  "path": "src/auth.py",
  "error": "File changed after it was last read.",
  "instruction": "Re-read this file with read_file before modifying it again."
}
```

这条 tool error 会进入对话，迫使模型重新 `read_file`，基于用户最新修改重新生成补丁。

### `memory/store.py`

MemoryStore 只管理 session memory：

```text
memory/handoff.md
memory/scratchpad.json
```

`scratchpad.json` 保存当前 session 的结构化工作状态；压缩后完整 carry forward 到新 session。

### `memory/stores/long_term.py`

`LongTermMemoryStore` 管理 conversation 级长期记忆候选：

```text
conversation/memory/raw/YYYY-MM-DD.jsonl
```

每条记录包含：

- `type`：`personal` / `procedural` / `knowledge`
- `content`：独立可读的一句话记忆
- `reason`：为什么值得长期保存
- `confidence`：0 到 1
- `source`：当前为 `context_compaction`
- `evidence`：归档 dialog 路径

长期记忆不默认进入 prompt；后续通过 `memory_search` 或 dream 整理再复用。

### `hooks/compaction_hook.py`

`pre_llm` 阶段：

1. 判断是否需要压缩。
2. 调用 `ContextCompressor.compress()`。
3. 如果压缩成功且存在 `SessionRuntime`，调用 `ConversationStore.compact_session()`。
4. 更新 `SessionRuntime.current` 为新 session。

### `hooks/memory_hook.py`

`after_tool` 阶段把工具结果投影到 scratchpad：

- `read_files`
- `modified_files`
- `known_issues`
- `last_verified_commands`

不再写 `tool_index.jsonl` 或 `file_summaries.json`。

## 压缩与回溯

```text
session A active
  │
  ├─ pre_llm 超限
  ├─ 旧消息归档到 session A/runs/<run_id>/agent_context/dialog
  ├─ 生成 compact summary + long-term memories
  ├─ memories 写入 conversation/memory/raw
  ├─ session A 标记 compacted
  ├─ 创建 session B
  ├─ session B 保存 summary + protected messages
  ├─ session B 继承 scratchpad
  └─ conversation.active_session_id -> session B
```

回溯时：

```text
session_search(query=...)
  -> 找到 agent_context/dialog、agent_context/tool_result、session、memory 路径
read_file(path=...)
  -> 读取原文
```

## 为什么这样设计

- 压缩前后不混在同一个 session。
- 旧 session 保存完整原始消息，新 session 从摘要继续。
- scratchpad 继承，避免丢失当前任务状态。
- active session 最后原子切换，避免中断产生悬空 session。
- memory 只保留必要状态，避免派生缓存污染长期记忆。
- `session_search` 默认限制在当前 conversation，避免跨任务污染。
- Checkpoint 放在 conversation 下，压缩切新 session 后仍能保护同一条任务线里的文件写入。
