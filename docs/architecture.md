# coding-agent 架构说明

`coding-agent` 的运行时按职责拆成五个域：

- **Orchestrator**：管理 run 生命周期和 LLM-Tool 循环。
- **Context**：管理结构化上下文、压缩和最终 messages 编译。
- **Execution**：执行工具调用，并用 sandbox/policy 限制副作用。
- **Memory**：保存 session 级 handoff 与 scratchpad。
- **Telemetry**：保存 run 工件、events、trace 和 debug payload。

设计原则：工具层只执行动作，Context 只保存数据，Memory 只存当前 session 的必要状态，Run artifacts 保存可审计原文。

## 存储模型

```text
.coding-agent/
  conversations/
    <conversation_id>/
      conversation.json
      sessions/
        <session_id>/
          session.json
          memory/
            handoff.md
            scratchpad.json
          runs/
            <run_id>/
              task_state.json
              report.json
              trace.jsonl
              events.jsonl
              debug/
              dialog/
                *.jsonl
              tool_result/
                *.txt
```

语义：

- `conversation`：长期任务线。
- `session`：上下文窗口 epoch。
- `run`：一次 AgentLoop 执行。

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
- 注册默认工具和 `session_search`
- 注入 `ContextCompactionHook` 和 `MemoryProjectionHook`
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
2. 归档旧消息到 `runs/<run_id>/dialog/*.jsonl`。
3. 用压缩模型生成结构化 Markdown handoff。
4. 写当前 memory handoff。
5. 将 compact summary 作为 assistant 消息插入 active history。
6. 返回 `CompressionResult` 给 hook 做 session rotation。

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
- 调用 `ToolRegistry`
- 触发 tool hooks
- 生成 OpenAI tool message
- 对超长 tool result 做执行期 offload

超长输出写到：

```text
session/runs/<run_id>/tool_result/*.txt
```

### `memory/store.py`

MemoryStore 只管理 session memory：

```text
memory/handoff.md
memory/scratchpad.json
```

`scratchpad.json` 保存当前 session 的结构化工作状态；压缩后完整 carry forward 到新 session。

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
  ├─ 旧消息归档到 session A/runs/<run_id>/dialog
  ├─ 生成 compact summary
  ├─ session A 标记 compacted
  ├─ 创建 session B
  ├─ session B 保存 summary + protected messages
  ├─ session B 继承 scratchpad
  └─ conversation.active_session_id -> session B
```

回溯时：

```text
session_search(query=...)
  -> 找到 dialog/tool_result/session/memory 路径
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
