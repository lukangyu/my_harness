# coding-agent

`coding-agent` 是一个面向本地代码仓库的 AI 编程助手 CLI。它通过 OpenAI-compatible `/chat/completions` 接口调用模型，把文件读写、文本搜索、补丁应用、命令执行和历史记忆搜索包装成受控工具，并用工作区沙箱、命令策略、上下文压缩和运行归档控制长任务的复杂度。

当前架构重点是边界清晰：

- 编排层只管理 run 生命周期和 LLM-Tool 循环。
- Context 层只管理结构化上下文和最终 messages 编译。
- Execution 层只执行工具，不写 memory，不写 telemetry。
- Memory 层保存 scratchpad、handoff、工具索引、对话归档和长工具输出。
- Telemetry 层记录 run 工件、事件和 trace。

完整设计见：[docs/architecture.md](docs/architecture.md)。

## 当前能力

- 可安装命令：`coding-agent`
- 项目配置：`.coding-agent/config.toml`
- 一次性任务：`coding-agent run "..."`
- 交互聊天：`coding-agent chat`
- 会话恢复：`coding-agent chat --resume PATH` 或 `--resume-latest`
- OpenAI-compatible Chat Completions
- 原生 Function Calling 工具数组，不在 system prompt 中重复列工具定义
- 内置工具：
  - `list_files`
  - `read_file`
  - `write_file`
  - `search_text`
  - `apply_patch`
  - `run_shell`
  - `session_search`
- `read_file` 支持 `start_line` / `end_line`，大文件默认 50KB 截断并返回续读行号
- `search_text` 和 `session_search` 优先使用 `rg`，不可用时回退 Python 搜索
- 工作区文件工具受 `WorkspaceSandbox` 限制
- Shell 命令受 allow/deny 策略和审批回调限制
- ContextFrame 结构化上下文管线
- pre-LLM 上下文压缩、dialog archive、assistant 摘要消息
- 压缩摘要写回 `.coding-agent/memory/handoff.md`
- 长工具输出在执行期 offload 到 `tool_result/*.txt`
- 工具结果投影到 scratchpad、tool index 和 file summaries
- run 级 telemetry、trace、events、debug payload
- sanitized session 保存到 `.coding-agent/sessions/`

## 项目结构

```text
src/coding_agent/
  cli.py            CLI 命令：init、run、chat
  config.py         TOML 配置读取、校验、环境变量解析
  application.py    运行时依赖装配，注册 hooks 和 session_search
  llm.py            OpenAI-compatible Chat Completions 客户端
  session.py        JSON session 保存与恢复
  run_result.py     run 返回对象
  runtime_events.py CLI/UI 运行事件

  orchestrator/     RunCoordinator、AgentLoop、生命周期 Hook
  context/          ContextFrame、ContextCompressor、PromptBuilder
  execution/        ToolRegistry、ToolExecutor、Sandbox、Shell、Policy
  memory/           Scratchpad、Handoff、ToolIndex、DialogArchive、ToolResult
  hooks/            ContextCompactionHook、MemoryProjectionHook
  telemetry/        RunStore、TelemetryLogger

tests/              单元测试与 CLI smoke tests
docs/               架构文档和设计计划
```

## 运行流程

```text
CLI
  -> Application 装配依赖
  -> RunCoordinator 创建 run 工件
  -> AgentLoop 构建 Context
  -> pre_llm hooks 检查并执行上下文压缩
  -> PromptBuilder 生成 messages
  -> LLM 返回 assistant 消息或 tool_calls
  -> ToolExecutor 执行工具并 offload 长结果
  -> after_tool hooks 投影记忆
  -> 循环直到最终答案或 max_steps
```

核心边界：

- `Application` 只负责装配。
- `RunCoordinator` 负责 run 生命周期和报告落盘。
- `AgentLoop` 负责 LLM-Tool 状态机。
- `ToolRegistry` 只负责工具注册和路由，不写 memory/telemetry。
- `Context` 只保存结构化 frame，不主动读取 MemoryStore。
- `PromptBuilder` 是唯一最终消息工厂。

## 上下文管理

每次请求模型时，消息按 KV Cache 友好的顺序构建：

```text
system: 固定规训和工具使用边界
user: workspace / scratchpad / handoff / file summaries
assistant/user/tool: active history
user: current task
```

`Context` 内部保存两条历史：

- `_raw_frames`：完整原始历史，不因压缩被破坏。
- `_active_frames`：当前送入模型的历史子集。

触发条件：

```text
estimate_active_tokens > max_input_tokens * compact_threshold_ratio
```

默认配置下是：

```text
24000 * 0.8 = 19200
```

触发后，`ContextCompactionHook` 在 `pre_llm` 阶段调用 `ContextCompressor`：

1. 按 `compact_tail_ratio` 和 `protected_recent_turns` 保护近场消息。
2. 对 tool_call / tool_result 边界做对齐，避免拆散工具调用组。
3. 将旧消息归档到 `.coding-agent/runs/<run_id>/dialog/*.jsonl`。
4. 使用压缩模型生成结构化 Markdown 摘要。
5. 把摘要写回 `.coding-agent/memory/handoff.md`。
6. 将摘要作为 `assistant` 消息插入 active history。

摘要形态：

```text
[CONTEXT COMPACTION] Earlier turns were compacted.

Archive: .coding-agent/runs/<run_id>/dialog/2026-05-24-xxxxxxxx.jsonl

## 目标
...

## 进度
...

## 下一步
...

Past conversation history has been archived. Use read_file on archive_log_path if cross-session facts are missing.
```

压缩不会清空真实历史。后续保存 session、写 trace 或排查问题时，仍可以从 session、run trace、dialog archive 找回原始内容。

## 记忆与回溯

运行数据主要落在三个位置：

```text
.coding-agent/memory/
  scratchpad.json       工具事实、已读/已改文件、问题、验证命令
  handoff.md            当前任务交接摘要，压缩后会更新
  tool_index.jsonl      工具调用索引和摘要
  file_summaries.json   文件结构摘要缓存

.coding-agent/sessions/
  *.json                可 resume 的 sanitized 对话

.coding-agent/runs/<run_id>/
  dialog/*.jsonl        被压缩归档的旧消息
  tool_result/*.txt     超长工具输出全文
  trace.jsonl
  events.jsonl
  debug/
```

`session_search` 是专门给模型回溯记忆的只读工具。它搜索：

- `.coding-agent/memory`
- `.coding-agent/sessions`
- `.coding-agent/runs`

支持参数：

- `query`: 搜索文本或正则
- `case_sensitive`: 默认 `false`
- `regex`: 默认 `false`
- `glob`: 可选文件 glob
- `max_matches`: 默认 `50`
- `sources`: 可选，`memory` / `sessions` / `runs` / `dialog` / `tool_result`

返回结果包含：

- `source`
- `path`
- `line`
- `text`
- `metadata.engine`: `rg` 或 `python`
- `metadata.searched_roots`
- `metadata.truncated`

当模型发现当前上下文里缺少历史决策、旧对话细节或长工具输出时，可以先用 `session_search` 定位，再用 `read_file` 读取具体归档文件。

## 工具结果 Offload

长工具输出不等到上下文压缩才处理。`ToolExecutor` 在生成 tool message 时会立即调用 `MemoryStore.offload_tool_result()`。

默认超过 4000 字符会写入：

```text
.coding-agent/runs/<run_id>/tool_result/*.txt
```

tool message 中只保留截断片段、路径和读取提示。这样下一轮 prompt 不会被大段终端输出或文件内容撑爆。

## 配置

初始化：

```bash
coding-agent init
```

示例：

```toml
[model]
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-4.1"

[agent]
max_steps = 20
stream = true

[workspace]
root = "."

[commands]
allow = [
  "python -m pytest",
  "pytest",
  "ruff",
  "mypy",
  "git status",
  "git diff"
]
deny = [
  "rm",
  "del",
  "rmdir",
  "git reset",
  "git checkout",
  "powershell Remove-Item"
]

[context]
max_input_tokens = 24000
compact_threshold_ratio = 0.8
compact_tail_ratio = 0.2
protected_recent_turns = 4
protected_tool_results = 6
doc_max_chars = 1200
tree_max_entries = 200
include_project_docs = true
include_file_tree = true
include_git_status = true
include_recent_commits = true
show_cache_stats = true
```

PowerShell 设置 API key：

```powershell
$env:OPENAI_API_KEY = "..."
```

## 命令

### `coding-agent init`

创建 `.coding-agent/config.toml`。

```bash
coding-agent init
coding-agent init --path /path/to/project
```

### `coding-agent run`

执行一次任务并退出。

```bash
coding-agent run "检查项目结构并总结"
coding-agent run "运行测试并解释失败原因"
```

### `coding-agent chat`

启动交互式会话。

```bash
coding-agent chat
coding-agent chat --resume .coding-agent/sessions/20260513-120000-000000.json
coding-agent chat --resume-latest
```

Slash commands：

- `/exit`：退出 chat
- `/clear`：清空当前对话历史
- `/status`：打印当前消息数量

## 内置工具

### `list_files`

列出工作区路径下的文件。

参数：

- `path = "."`
- `max_entries = 200`
- `max_depth`

### `read_file`

读取 UTF-8 文件。

参数：

- `path`
- `start_line = 1`
- `end_line`
- `max_chars = 50000`

大文件会自动截断，并在 metadata 中返回：

- `total_lines`
- `end_line`
- `next_start_line`
- `notice`

### `write_file`

写入 UTF-8 文件，可自动创建父目录。

### `search_text`

搜索工作区文件。优先使用 `rg`，不可用时回退 Python 遍历。

参数：

- `query`
- `path = "."`
- `case_sensitive = true`
- `regex = false`
- `glob`
- `max_matches = 100`

### `session_search`

搜索 Agent 记忆、历史 session、run 归档和长工具输出。它不是普通工作区搜索，而是专门给上下文回溯使用。

参数：

- `query`
- `case_sensitive = false`
- `regex = false`
- `glob`
- `max_matches = 50`
- `sources`

### `apply_patch`

应用 OpenAI-style patch，支持 add/delete/update/move。

### `run_shell`

通过 `CommandPolicy` 和 `ShellRunner` 执行命令。默认 `shell=False`，拒绝 shell 控制操作符和未授权命令。

## 安全模型

### 工作区沙箱

所有文件工具都通过 `WorkspaceSandbox` 解析路径：

- 相对路径基于 `workspace.root`
- 绝对路径也必须在 workspace 内
- `../outside.txt` 会被拒绝

### 命令策略

Shell 命令执行前会经过：

1. deny 规则
2. shell 控制操作符检查
3. allow 规则
4. 审批回调

未命中 allow 的命令会被拒绝。

## 开发

安装开发依赖：

```bash
python -m pip install -e ".[dev]"
```

运行测试：

```bash
python -m pytest -q
```

本仓库常用验证命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.tmp-pytest
```

## 当前限制

- token 预算目前使用近似字符估算，不是模型 tokenizer。
- `session_search` 是文本/正则搜索，还没有语义向量检索。
- 压缩摘要质量取决于当前配置的模型。
- 当前 CLI 仍是简单命令行界面，没有全屏 TUI。
