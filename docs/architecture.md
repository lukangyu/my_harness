# coding-agent 架构说明

`coding-agent` 是一个面向本地代码仓库的 AI 编程助手。它把一次任务拆成四条清晰的运行管线：

- **Orchestrator**：控制一次 run 的生命周期和 LLM-Tool 循环。
- **Context**：把工作区、记忆、历史消息和当前任务组装成模型请求。
- **Execution**：执行模型请求的工具调用，并把文件与命令能力限制在安全边界内。
- **Memory / Telemetry**：沉淀运行事实、归档长上下文，并记录可追踪的运行工件。

设计目标不是把所有逻辑塞进一个 Agent 类，而是让每个域只处理自己的状态和副作用。这样工具层可以保持纯执行，Context 可以保持纯数据，长期记忆和遥测也不会反向污染工具实现。

## 总览

```text
src/coding_agent/
├─ cli.py                 # CLI 入口：init/run/chat、Rich 输出、会话恢复
├─ config.py              # 配置读取与校验：.coding-agent/config.toml
├─ application.py         # 运行时装配：依赖注入、Hook 注册、组件拼装
├─ llm.py                 # OpenAI-compatible Chat Completions 客户端
├─ session.py             # 会话 JSON 存取，保存可恢复的对话消息
├─ run_result.py          # run 返回对象
├─ runtime_events.py      # 面向 UI/CLI 的轻量运行事件
│
├─ orchestrator/          # 生命周期与编排域
├─ context/               # 上下文与提示词域
├─ execution/             # 工具执行与安全边界域
├─ memory/                # 长期记忆与持久化域
├─ hooks/                 # 生命周期 Hook 实现
└─ telemetry/             # 运行工件与观测域
```

## 一次任务如何运行

```text
CLI
  │
  ▼
Application
  ├─ 创建 WorkspaceSandbox / CommandPolicy / ShellRunner
  ├─ 创建 ToolRegistry
  ├─ 创建 MemoryStore / RunStore / TelemetryLogger
  ├─ 创建 ContextAssembler / PromptBuilder
  └─ 注入 ContextCompactionHook / MemoryProjectionHook
      │
      ▼
RunCoordinator
  ├─ 创建 run 目录
  ├─ 记录 task_state / report
  └─ 调用 AgentLoop
      │
      ▼
AgentLoop
  ├─ ContextAssembler 构建 Context
  ├─ pre_llm hooks：必要时压缩历史
  ├─ PromptBuilder 生成 messages
  ├─ LLM 调用
  ├─ ToolExecutor 执行工具调用
  └─ 循环直到最终答案或 max_steps
```

这条路径里，`Application` 只负责装配，`RunCoordinator` 只负责 run 生命周期，`AgentLoop` 只负责 LLM-Tool 状态机。业务副作用通过 Hook 和 Store 处理。

## 核心模块职责

### `cli.py`

人类交互入口。负责：

- `coding-agent init`
- `coding-agent run`
- `coding-agent chat`
- chat 模式下的 `/exit`、`/clear`、`/status`
- 加载历史 session
- 打印最终答案、缓存命中率、session/run 路径

CLI 不直接知道工具如何执行，也不直接构造 prompt。它只把用户输入交给 `Application`。

### `config.py`

配置层。负责读取 `.coding-agent/config.toml`，校验模型、工作区、命令策略和上下文预算配置。

关键配置包括：

- `model.base_url`
- `model.api_key_env`
- `model.model`
- `agent.max_steps`
- `workspace.root`
- `commands.allow`
- `commands.deny`
- `context.max_input_tokens`
- `context.compact_threshold_ratio`

### `application.py`

装配层。它是运行时依赖注入的集中位置。

负责创建并连接：

- `WorkspaceSandbox`
- `CommandPolicy`
- `ShellRunner`
- `ToolRegistry`
- `OpenAICompatibleClient`
- `RunStore`
- `TelemetryLogger`
- `MemoryStore`
- `ContextAssembler`
- `PromptBuilder`
- `RunCoordinator`
- `AgentLoop`
- lifecycle hooks

设计原因：把依赖关系集中到这里，可以避免底层模块互相 import 对方的实现。例如 `Context` 不需要知道 `MemoryStore`，`ToolRegistry` 不需要知道 telemetry。

## Orchestrator 域

### `orchestrator/coordinator.py`

`RunCoordinator` 管理单次 run 的外层生命周期。

它负责：

- 开始 run 工件目录
- 写 `task_state.json`
- 写 `report.json`
- 保存 session
- 记录 workspace 前后快照
- 捕获中断与异常状态

它不负责工具执行，也不负责 prompt 组装。

### `orchestrator/agent_loop.py`

`AgentLoop` 是纯 LLM-Tool 状态机。

每一轮执行：

1. 创建 `AgentTurnContext`
2. 触发 `on_turn_start`
3. 触发 `pre_llm`
4. 用 `PromptBuilder` 生成最终 `messages`
5. 调用 LLM
6. 触发 `after_llm`
7. 如果有 tool calls，交给 `ToolExecutor`
8. 把 tool message 加回当前轮上下文
9. 无工具调用时返回最终答案

它的边界是：控制循环，但不把 memory/telemetry 逻辑写进工具函数。

### `orchestrator/lifecycle.py`

定义 Hook 协议和 `AgentLifecycleBus`。

当前支持的阶段：

- `on_turn_start`
- `pre_llm`
- `after_llm`
- `pre_tool`
- `after_tool`
- `on_turn_end`

设计原因：压缩、记忆投影、遥测扩展都可以挂在生命周期上，不需要改 AgentLoop 主流程。

## Context 域

### `context/context.py`

`Context` 是上下文状态实体。它保存结构化 frame，而不是直接拼最终 prompt 文本。

主要对象：

- `UsageStats`
- `WorkspaceContextOptions`
- `WorkspaceSnapshot`
- `ContextFrame`
- `Context`
- `estimate_tokens`

`ContextFrame` 是上下文管线的最小单元：

```python
ContextFrame(
    kind="workspace",
    role=None,
    payload={...},
    priority=90,
    stability="medium",
    token_estimate=1200,
)
```

当前 frame 大致分为：

- `workspace`
- `memory`
- `handoff`
- `file_summaries`
- `history`
- `compact_summary`
- `current_task`

压缩后，`compact_summary` 会作为 `assistant` 消息进入历史流：

```text
[CONTEXT COMPACTION] Earlier turns were compacted.

Archive: dialog/2026-05-22-xxxx.jsonl

## 目标
...
```

`Context` 同时保留：

- `_raw_frames`：完整原始历史，用于保留真实对话流。
- `_active_frames`：当前要喂给模型的历史子集。

这样压缩不会破坏原始对话历史。

### `context/assembler.py`

`ContextAssembler` 负责从外部 Store 读取原始数据，并 push 进 `Context`。

它读取：

- workspace snapshot
- scratchpad
- handoff
- file summaries
- prior messages
- tool schemas

设计原因：`Context` 不主动依赖 `MemoryStore`。数据流是 Push，不是 Pull。

### `context/prompt_builder.py`

`PromptBuilder` 是唯一的最终消息工厂。

它负责：

- 管理三层系统规训：
  - system base
  - tool guidance
  - tool enforcement
- 根据 `Context` 输出 OpenAI-compatible `list[dict]`
- 保持 KV Cache 友好的消息顺序
- 支持 JSON/XML 包装策略

输出顺序：

```text
system: 固定系统规训
user: workspace / memory / handoff / file summaries
assistant/user/tool: active history
user: current task
```

工具 schema 不会重复写入 system prompt。工具定义通过 API 的 `tools` 数组传递；system prompt 只写工具使用边界，例如不要调用不存在的工具。

## Execution 域

### `execution/tools.py`

`ToolRegistry` 只负责注册和路由工具。

默认工具：

- `list_files`
- `read_file`
- `write_file`
- `search_text`
- `apply_patch`
- `run_shell`

工具函数只执行物理动作并返回结果，不写 memory，不写 telemetry。

`read_file` 支持：

- `start_line`
- `end_line`
- `max_chars = 50000`
- 截断时返回 `next_start_line`

设计原因：未来接入 MCP 或远程工具时，不能要求工具函数自己知道本地 memory/telemetry。

### `execution/executor.py`

`ToolExecutor` 是模型 tool call 和本地工具之间的适配层。

负责：

- 解析 JSON arguments
- 触发 `pre_tool`
- 调用 `ToolRegistry`
- 触发 `after_tool`
- 生成 OpenAI tool message
- 对超长 tool result 做执行期 Offload

长工具结果默认超过 4000 字符会写入：

```text
.coding-agent/runs/<run_id>/tool_result/*.txt
```

tool message 中只保留片段和路径引用。

### `execution/sandbox.py`

工作区路径沙箱。所有文件工具都必须通过它解析路径。

它保证：

- 相对路径基于 workspace root
- 绝对路径也必须在 workspace root 内
- `../outside.txt` 这类越界路径会被拒绝

### `execution/policy.py`

命令策略。

规则：

- deny 优先
- allow 后置
- 未命中 allow 则拒绝
- shell 控制操作符会被拒绝

### `execution/shell.py`

命令执行器。

负责：

- 调用 `CommandPolicy`
- 使用 `shell=False`
- 捕获 stdout/stderr/exit code/timeout
- 拒绝执行 workspace-local 可执行文件
- 支持命令审批回调

## Memory 域

### `memory/store.py`

`MemoryStore` 是 memory 域的 Facade，聚合多个小 Store。

它提供：

- scratchpad 读写
- handoff 读写
- dialog archive
- tool result offload
- tool index append
- file summary load/save/update/invalidate
- tool result 投影入口

它不拼最终 prompt。prompt 文本化属于 `PromptBuilder`。

### `memory/projector.py`

`ToolMemoryProjector` 把工具执行结果转成长期状态更新。

例如：

- `read_file` 成功后更新 `read_files`
- `write_file` 成功后更新 `modified_files`
- `apply_patch` 成功后标记文件摘要 stale
- `run_shell` 记录最近验证命令
- 失败结果写入 `known_issues`
- 每次工具调用追加 `tool_index.jsonl`

设计原因：工具函数保持纯净，memory 更新通过 hook/projector 发生。

### `memory/stores/scratchpad.py`

结构化任务便签。

默认字段：

- `project_goal`
- `user_preferences`
- `confirmed_decisions`
- `modified_files`
- `read_files`
- `known_issues`
- `active_todos`
- `last_verified_commands`

当前主要由工具投影器写入工具事实。

### `memory/stores/handoff.py`

跨 run 交接摘要文件：

```text
.coding-agent/memory/handoff.md
```

设计用途是保存任务目标、关键决策、进度、下一步。当前读取链路已经接入 Context，写入链路还可以继续加强。

### `memory/stores/dialog_archive.py`

保存被压缩掉的旧历史消息。

当前 run 内路径：

```text
.coding-agent/runs/<run_id>/dialog/YYYY-MM-DD-xxxxxxxx.jsonl
```

### `memory/stores/tool_result.py`

保存超长工具输出。

路径：

```text
.coding-agent/runs/<run_id>/tool_result/*.txt
```

### `memory/stores/tool_index.py`

记录工具调用摘要：

```text
.coding-agent/memory/tool_index.jsonl
```

适合快速了解之前执行过什么工具，但不是完整对话记忆。

### `memory/stores/file_summary.py`

代码文件摘要缓存。

保存：

- 文件 hash
- language
- imports
- symbols
- heading/function/class 预览

用于给上下文提供轻量代码结构信息。

## Hooks 域

### `hooks/compaction_hook.py`

`ContextCompactionHook` 在每轮 LLM 前运行。

触发条件：

```text
estimate_active_tokens > max_input_tokens * compact_threshold_ratio
```

默认约为：

```text
24000 * 0.8 = 19200
```

触发后：

1. 切分旧历史和近场保留区
2. 旧历史写入 `dialog/*.jsonl`
3. 调用轻量摘要 LLM
4. 生成 `[CONTEXT COMPACTION]` assistant 摘要消息
5. active history 变成：

```text
assistant compact summary
protected recent messages
current task
```

工具结果 Offload 不在这里做，它属于 `ToolExecutor` 的执行期逻辑。

### `hooks/memory_hook.py`

`MemoryProjectionHook` 在 `after_tool` 阶段运行。

职责是把工具执行结果交给 `MemoryStore.record_tool_result()`，从而更新 scratchpad、tool index、file summaries 等。

## Telemetry 域

### `telemetry/store.py`

`RunStore` 管理 run 目录：

```text
.coding-agent/runs/<run_id>/
├─ task_state.json
├─ report.json
├─ trace.jsonl
├─ events.jsonl
├─ debug/
├─ dialog/
└─ tool_result/
```

### `telemetry/logger.py`

`TelemetryLogger` 写入：

- `events.jsonl`
- `trace.jsonl`
- workspace snapshot
- span duration
- model/tool/run 阶段事件

它服务于排查问题，不参与业务决策。

## Session 与 Memory 的区别

`session.py` 保存可恢复对话：

```text
.coding-agent/sessions/*.json
```

它保存的是 sanitized conversation messages，不包含生成的 workspace context 或 current task wrapper。

Memory 则保存长期状态：

```text
.coding-agent/memory/
├─ scratchpad.json
├─ handoff.md
├─ tool_index.jsonl
├─ file_summaries.json
└─ ...
```

当前系统里，session 是完整对话恢复入口；memory 更偏运行事实和上下文辅助。后续如果要做 `session_search`，应该把 session/dialog/decision/handoff 都纳入搜索源。

## 为什么这样设计

### 1. 工具层保持纯净

工具只执行动作并返回数据。它不写 memory，不写 telemetry。

这样未来替换成 MCP、远程工具或外部工具服务时，不需要修改工具源码来适配本地记忆系统。

### 2. Context 不依赖 MemoryStore

`Context` 只保存结构化 frame。外部由 `ContextAssembler` 把 memory 数据推入。

这样可以独立测试 Context 的切分、压缩和消息输出，而不用 mock 整个 MemoryStore。

### 3. PromptBuilder 是唯一消息工厂

Context 不拼 XML/Markdown，最终字符串化集中在 PromptBuilder。

这样未来可以按模型切换 JSON/XML 包装策略，同时不污染 Context 数据结构。

### 4. 压缩通过 Hook 发生

压缩是运行时策略，不是 Context 自身副作用。

`pre_llm` 阶段做压缩，可以保证每次请求模型前都有机会控制上下文窗口，同时保留原始历史。

### 5. 工具结果 Offload 在执行期完成

超长工具输出在 tool message 生成时就落盘，不等到下一轮压缩再处理。

这样 prompt 中不会堆积巨型 tool result，ContextCompactionHook 也不需要理解工具输出格式。

## 当前边界和后续方向

已经落地：

- 模块化 domain package
- ContextFrame 结构化上下文
- PromptBuilder 三层规训
- lifecycle hook
- 对话压缩和 dialog archive
- tool result offload
- 工具结果 memory projection
- telemetry/run artifacts

仍可继续增强：

- 把 compaction summary 写回 `handoff.md`
- 增加 `session_search`
- 增加 decision log
- 把 session history 投影到 memory 搜索索引
- 更精确的 tokenizer 预算
- 更结构化的 compact summary payload
- 更多手动诊断命令，例如 `/compact`、`/dump_context`
