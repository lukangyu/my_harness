# coding-agent 架构说明

`coding-agent` 是一个面向本地代码仓库的 AI 编程助手。它的核心不是一个巨大的 Agent 类，而是几条职责清晰的管线：

- **Orchestrator**：管理 run 生命周期和 LLM-Tool 状态机。
- **Context**：管理结构化上下文、压缩和最终 messages 编译。
- **Execution**：执行工具调用，并把文件和命令副作用限制在安全边界内。
- **Memory**：保存任务事实、交接摘要、历史归档、工具索引和长输出。
- **Telemetry**：记录可排查的 run 工件、events、trace 和 debug payload。

这种拆分的目标是让每个模块只处理自己的状态和副作用。工具层不写 memory，Context 不主动拉取 Store，PromptBuilder 不执行工具，Telemetry 不参与业务决策。

## 总览

```text
src/coding_agent/
├─ cli.py                 # CLI 入口：init/run/chat、Rich 输出、会话恢复
├─ config.py              # 配置读取与校验：.coding-agent/config.toml
├─ application.py         # 运行时装配：依赖注入、Hook 注册、工具注册
├─ llm.py                 # OpenAI-compatible Chat Completions 客户端
├─ session.py             # 会话 JSON 存取，保存可恢复的对话消息
├─ run_result.py          # run 返回对象
├─ runtime_events.py      # 面向 UI/CLI 的轻量运行事件
│
├─ orchestrator/          # 生命周期与编排域
├─ context/               # 上下文、压缩器、提示词构建域
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
  ├─ 创建 ToolRegistry，并注册默认工具和 session_search
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
  ├─ after_tool hooks：投影记忆
  └─ 循环直到最终答案或 max_steps
```

这条链路里：

- `Application` 只负责装配。
- `RunCoordinator` 只负责 run 外层生命周期。
- `AgentLoop` 只负责 LLM-Tool 循环。
- `ContextCompactionHook` 只负责 pre-LLM 压缩策略。
- `MemoryProjectionHook` 只负责工具结果到 memory 的投影。

## 核心模块职责

### `cli.py`

人类交互入口。负责：

- `coding-agent init`
- `coding-agent run`
- `coding-agent chat`
- chat 模式下的 `/exit`、`/clear`、`/status`
- 加载历史 session
- 打印最终答案、缓存命中率、session/run 路径和工具事件

CLI 不执行工具，不构造 prompt，也不直接写 memory。

### `config.py`

配置层。负责读取 `.coding-agent/config.toml`，校验模型、工作区、命令策略和上下文预算。

关键配置：

- `model.base_url`
- `model.api_key_env`
- `model.model`
- `agent.max_steps`
- `workspace.root`
- `commands.allow`
- `commands.deny`
- `context.max_input_tokens`
- `context.compact_threshold_ratio`
- `context.compact_tail_ratio`
- `context.protected_recent_turns`
- `context.protected_tool_results`

### `application.py`

运行时依赖注入中心。

负责创建并连接：

- `WorkspaceSandbox`
- `CommandPolicy`
- `ShellRunner`
- `ToolRegistry`
- `session_search` 搜索根目录
- `OpenAICompatibleClient`
- `RunStore`
- `TelemetryLogger`
- `MemoryStore`
- `ContextAssembler`
- `PromptBuilder`
- `RunCoordinator`
- `AgentLoop`
- lifecycle hooks

`session_search` 在这里注册，而不是放进默认工具构造函数里，是为了保持 `create_default_tools()` 不依赖 memory/telemetry。工具注册表仍然只是工具注册和路由容器。

## Orchestrator 域

### `orchestrator/coordinator.py`

`RunCoordinator` 管理单次 run 的外层生命周期。

职责：

- 开始 run 工件目录
- 写 `task_state.json`
- 写 `report.json`
- 保存 session
- 记录 workspace 前后快照
- 捕获中断与异常状态

它不执行工具，也不构造 prompt。

### `orchestrator/agent_loop.py`

`AgentLoop` 是 LLM-Tool 状态机。

每一轮：

1. 创建 `AgentTurnContext`
2. 触发 `on_turn_start`
3. 触发 `pre_llm`
4. 用 `PromptBuilder` 生成最终 `messages`
5. 调用 LLM
6. 触发 `after_llm`
7. 如有 tool calls，交给 `ToolExecutor`
8. 把 tool message 加回当前轮上下文
9. 无工具调用时返回最终答案

它的边界是控制循环，不把 memory、telemetry、工具实现细节写进主流程。

### `orchestrator/lifecycle.py`

定义 Hook 协议和 `AgentLifecycleBus`。

阶段：

- `on_turn_start`
- `pre_llm`
- `after_llm`
- `pre_tool`
- `after_tool`
- `on_turn_end`

压缩、记忆投影、遥测扩展都通过 Hook 接入，不需要改 AgentLoop 主循环。

## Context 域

### `context/context.py`

`Context` 是上下文状态实体。它保存结构化 `ContextFrame`，不直接拼最终 prompt 文本。

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

当前 frame 类型：

- `workspace`
- `memory`
- `handoff`
- `file_summaries`
- `history`
- `compact_summary`
- `current_task`

`Context` 同时保存两条历史：

- `_raw_frames`：完整原始历史，不因压缩被破坏。
- `_active_frames`：当前要喂给模型的历史子集。

压缩后，`compact_summary` 作为 `assistant` 消息进入 active history：

```text
[CONTEXT COMPACTION] Earlier turns were compacted.

Archive: .coding-agent/runs/<run_id>/dialog/2026-05-24-xxxxxxxx.jsonl

## 目标
...
```

### `context/assembler.py`

`ContextAssembler` 负责从外部 Store 读取原始数据，并 push 进 `Context`。

读取：

- workspace snapshot
- scratchpad
- handoff
- file summaries
- prior messages
- tool schemas

设计原因：`Context` 不主动依赖 `MemoryStore`。数据流是 Push，不是 Pull。这样测试 Context 切分和压缩边界时，不需要 mock 整个 memory 层。

### `context/compressor.py`

`ContextCompressor` 是实际压缩引擎，由 `ContextCompactionHook` 在 `pre_llm` 阶段调用。

触发判断：

```text
context.estimate_active_tokens() > max_input_tokens * compact_threshold_ratio
```

压缩流程：

1. 根据 `compact_tail_ratio` 计算近场 token 预算。
2. 调用 `Context.slice_old_history()` 切出旧历史和保留区。
3. 保护最近用户轮次和工具结果数量。
4. 对 tool_call / tool_result 边界做对齐，避免 orphan tool result。
5. 调用 `MemoryStore.archive_dialog_messages()` 保存旧消息。
6. 构造包含 `previous_handoff`、`scratchpad`、`source_refs` 和 `old_messages` 的摘要请求。
7. 调用压缩模型生成结构化 Markdown。
8. 写回 `.coding-agent/memory/handoff.md`。
9. 调用 `Context.replace_active_history()` 插入 assistant 摘要。

摘要模板固定包含：

- `## 目标`
- `## 约束与偏好`
- `## 进度`
- `### 已完成`
- `### 进行中`
- `### 阻塞项`
- `## 关键决策`
- `## 相关文件`
- `## 下一步`
- `## 关键上下文`

### `context/prompt_builder.py`

`PromptBuilder` 是唯一最终消息工厂。

它负责：

- 管理三层系统规训：
  - system base
  - tool guidance
  - tool enforcement
- 根据 `Context` 输出 OpenAI-compatible `list[dict]`
- 保持 KV Cache 友好的消息顺序
- 支持 JSON/XML 包装策略
- 根据当前 tool schema 动态提示 `memory` / `session_search` 是否可用

输出顺序：

```text
system: 固定系统规训
user: workspace / memory / handoff / file summaries
assistant/user/tool: active history
user: current task
```

工具定义不写进 system prompt。具体工具 schema 通过 API 的 `tools` 数组传递；system prompt 只保留工具使用边界，例如不要调用当前 tools 数组中不存在的工具。

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

额外注册工具：

- `session_search`

工具函数只执行物理动作并返回结果，不写 memory，不写 telemetry。

#### `read_file`

读取 UTF-8 文件。

参数：

- `path`
- `start_line = 1`
- `end_line`
- `max_chars = 50000`

截断时返回：

- `total_lines`
- `end_line`
- `next_start_line`
- `notice`

#### `search_text`

搜索工作区文件。优先使用 `rg`，不可用时回退 Python 遍历。

#### `session_search`

搜索 Agent 记忆和历史工件。它不是普通代码搜索，而是给模型在上下文不清晰时回溯用。

搜索根由 `Application` 注入：

```text
.coding-agent/memory
.coding-agent/sessions
.coding-agent/runs
```

参数：

- `query`
- `case_sensitive = false`
- `regex = false`
- `glob`
- `max_matches = 50`
- `sources`: `memory` / `sessions` / `runs` / `dialog` / `tool_result`

返回：

```json
{
  "ok": true,
  "matches": [
    {
      "source": "dialog",
      "path": ".coding-agent/runs/<run_id>/dialog/2026-05-24-xxxxxxxx.jsonl",
      "line": 7,
      "text": "..."
    }
  ],
  "metadata": {
    "engine": "rg",
    "returned_matches": 1,
    "truncated": false,
    "searched_roots": [".coding-agent/runs"]
  }
}
```

`session_search` 的 root 是否存在是在调用时判断，因此运行过程中后续生成的 memory/run 文件也能被搜到。

### `execution/executor.py`

`ToolExecutor` 是模型 tool call 和本地工具之间的适配层。

职责：

- 解析 JSON arguments
- 触发 `pre_tool`
- 调用 `ToolRegistry`
- 触发 `after_tool`
- 生成 OpenAI tool message
- 对超长 tool result 做执行期 Offload
- 发出 runtime event 和 telemetry event

长工具结果默认超过 4000 字符会写入：

```text
.coding-agent/runs/<run_id>/tool_result/*.txt
```

tool message 中只保留片段和路径引用。这样下一轮 prompt 不会直接携带大段终端输出、文件内容或搜索结果。

### `execution/sandbox.py`

工作区路径沙箱。所有文件工具都必须通过它解析路径。

保证：

- 相对路径基于 workspace root
- 绝对路径也必须在 workspace root 内
- `../outside.txt` 会被拒绝

### `execution/policy.py`

命令策略。

规则：

- deny 优先
- 拒绝 shell 控制操作符
- allow 后置
- 未命中 allow 则拒绝

### `execution/shell.py`

命令执行器。

职责：

- 调用 `CommandPolicy`
- 使用 `shell=False`
- 捕获 stdout/stderr/exit code/timeout
- 拒绝执行 workspace-local 可执行文件
- 支持命令审批回调

## Memory 域

### `memory/store.py`

`MemoryStore` 是 memory 域 Facade，聚合多个小 Store。

提供：

- scratchpad 读写
- handoff 读写
- dialog archive
- tool result offload
- tool index append
- file summary load/save/update/invalidate
- tool result 投影入口

它不拼最终 prompt。面向模型的字符串化属于 `PromptBuilder`。

### `memory/projector.py`

`ToolMemoryProjector` 把工具执行结果转成长期状态更新。

示例：

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

### `memory/stores/handoff.py`

跨 run 交接摘要：

```text
.coding-agent/memory/handoff.md
```

压缩触发后，`ContextCompressor` 会把新的结构化摘要写回这个文件。下一轮 ContextAssembler 会把它作为 `handoff` frame 注入上下文。

### `memory/stores/dialog_archive.py`

保存被压缩掉的旧历史消息。

路径：

```text
.coding-agent/runs/<run_id>/dialog/YYYY-MM-DD-xxxxxxxx.jsonl
```

这些归档可通过 `session_search` 定位，再用 `read_file` 读取。

### `memory/stores/tool_result.py`

保存超长工具输出。

路径：

```text
.coding-agent/runs/<run_id>/tool_result/*.txt
```

这些文件同样可通过 `session_search` 搜索。

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

它现在很薄，只负责：

1. 创建并持有 `ContextCompressor`
2. `pre_llm` 时判断是否需要压缩
3. 需要时调用 `ContextCompressor.compress(ctx.context_entity)`

它不处理长工具输出。长工具输出属于执行期 offload。

### `hooks/memory_hook.py`

`MemoryProjectionHook` 在 `after_tool` 阶段运行。

职责是把工具执行结果交给：

```python
MemoryStore.record_tool_result(...)
```

然后由 `ToolMemoryProjector` 更新 scratchpad、tool index、file summaries 等。

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

## Session、Memory、Archive 的区别

### Session

路径：

```text
.coding-agent/sessions/*.json
```

保存可恢复聊天记录。它保存 sanitized conversation messages，不包含生成的 workspace context 或 current task wrapper。

### Memory

路径：

```text
.coding-agent/memory/
├─ scratchpad.json
├─ handoff.md
├─ tool_index.jsonl
├─ file_summaries.json
└─ ...
```

保存长期任务事实、交接摘要、工具索引和文件摘要。

### Archive

路径：

```text
.coding-agent/runs/<run_id>/dialog/*.jsonl
.coding-agent/runs/<run_id>/tool_result/*.txt
```

保存因为上下文预算或工具输出长度而离开 prompt 的原始内容。Archive 不会自动塞回上下文，需要模型通过 `session_search` / `read_file` 主动回溯。

## 上下文压缩与回溯链路

```text
AgentLoop 开始新一轮
  │
  ▼
pre_llm: ContextCompactionHook
  │
  ├─ estimate_active_tokens 未超限
  │    └─ 直接放行
  │
  └─ estimate_active_tokens 超限
       ├─ Context.slice_old_history()
       ├─ MemoryStore.archive_dialog_messages()
       ├─ compact LLM 生成 Markdown handoff
       ├─ MemoryStore.write_handoff()
       └─ Context.replace_active_history()
  │
  ▼
PromptBuilder.build_final_messages()
  │
  ▼
LLM
  │
  ├─ 当前上下文足够：继续任务
  │
  └─ 需要旧细节
       ├─ session_search(query=...)
       └─ read_file(path=archive_path, start_line=...)
```

压缩后，模型在 active history 中能看到 assistant 摘要和 archive path。需要更细节时，它可以先搜索 `.coding-agent` 记忆目录，再读取具体文件。

## 为什么这样设计

### 1. 工具层保持纯净

工具只执行动作并返回数据。它不写 memory，不写 telemetry。

这样未来替换成 MCP、远程工具或外部工具服务时，不需要修改工具源码来适配本地记忆系统。

### 2. Context 不依赖 MemoryStore

`Context` 只保存结构化 frame。外部由 `ContextAssembler` 把 memory 数据推入。

这样可以独立测试 Context 的切分、压缩边界和消息输出，而不用 mock 整个 MemoryStore。

### 3. PromptBuilder 是唯一消息工厂

Context 不拼 XML/Markdown，最终字符串化集中在 PromptBuilder。

这样未来可以按模型切换 JSON/XML 包装策略，同时不污染 Context 数据结构。

### 4. 压缩通过 Hook 发生

压缩是运行时策略，不是工具执行逻辑，也不是 Context 自己的副作用。

`pre_llm` 阶段做压缩，可以保证每次请求模型前都有机会控制上下文窗口，同时保留原始历史。

### 5. 工具结果 Offload 在执行期完成

超长工具输出在 tool message 生成时就落盘，不等到下一轮压缩再处理。

这样 prompt 中不会堆积巨型 tool result，ContextCompactionHook 也不需要理解工具输出格式。

### 6. session_search 只做回溯，不做记忆写入

`session_search` 是只读工具。它搜索 memory、session 和 run archive，但不更新任何状态。

这保持了工具层纯净，也让模型能够在压缩后主动找回历史细节。

## 当前边界和后续方向

已经落地：

- 模块化 domain package
- ContextFrame 结构化上下文
- PromptBuilder 三层规训
- lifecycle hook
- 对话压缩和 dialog archive
- compaction summary 写回 handoff
- compact summary 作为 assistant 消息进入 active history
- tool result 执行期 offload
- 工具结果 memory projection
- `session_search` 记忆搜索工具
- telemetry/run artifacts

仍可继续增强：

- 更精确的 tokenizer 预算
- `session_search` 的语义检索或索引缓存
- decision log 和用户偏好抽取
- 更结构化的 compact summary payload
- 手动诊断命令，例如 `/compact`、`/dump_context`
