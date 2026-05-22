# coding-agent

`coding-agent` 是一个面向本地代码仓库的 AI 编程助手 CLI。它通过 OpenAI-compatible `/chat/completions` 接口调用模型，把文件读写、文本搜索、补丁应用和命令执行包装成受控工具，并用明确的工作区沙箱和命令策略约束所有副作用。

项目当前重点是把 Agent 运行时拆成清晰的领域模块：编排、上下文、工具执行、记忆、观测分别负责自己的边界，避免工具函数、Prompt 组装和持久化逻辑互相穿透。

完整架构说明见：[docs/architecture.md](docs/architecture.md)。

## 当前能力

- 可安装命令：`coding-agent`
- 项目本地配置：`.coding-agent/config.toml`
- 一次性任务：`coding-agent run "..."`
- 交互式聊天：`coding-agent chat`
- 从保存的 session 恢复聊天：`coding-agent chat --resume PATH` 或 `--resume-latest`
- OpenAI-compatible Chat Completions 调用
- 内置工具：
  - `list_files`
  - `read_file`
  - `write_file`
  - `search_text`
  - `apply_patch`
  - `run_shell`
- 文件工具路径受 `WorkspaceSandbox` 限制
- Shell 命令受 allow/deny 策略和审批回调限制
- ContextFrame 结构化上下文管线
- pre-LLM 上下文压缩与 dialog archive
- 长工具输出执行期 offload
- 工具结果投影到 memory scratchpad / tool index
- run 级 telemetry、trace、events、debug payload
- sanitized session 保存到 `.coding-agent/sessions/`

## 项目结构

```text
src/coding_agent/
  cli.py            CLI 命令：init、run、chat
  config.py         TOML 配置读取、校验、环境变量解析
  application.py    运行时依赖装配
  llm.py            OpenAI-compatible Chat Completions 客户端
  session.py        JSON session 保存与恢复
  run_result.py     run 返回对象
  runtime_events.py CLI/UI 运行事件

  orchestrator/     RunCoordinator、AgentLoop、生命周期 Hook
  context/          ContextFrame、WorkspaceSnapshot、PromptBuilder
  execution/        ToolRegistry、ToolExecutor、Sandbox、Shell、Policy
  memory/           Scratchpad、Handoff、ToolIndex、DialogArchive、FileSummary
  hooks/            ContextCompactionHook、MemoryProjectionHook
  telemetry/        RunStore、TelemetryLogger

tests/              单元测试与 CLI smoke tests
docs/               架构文档和设计计划
```

## 运行流程

一次 `coding-agent run` 或 chat 中的一轮请求大致如下：

```text
CLI
  -> Application 装配依赖
  -> RunCoordinator 创建 run 工件
  -> AgentLoop 构建 Context
  -> pre_llm hooks 检查是否需要压缩
  -> PromptBuilder 生成 messages
  -> LLM 返回 assistant 消息或 tool_calls
  -> ToolExecutor 执行工具
  -> after_tool hooks 投影记忆
  -> 循环直到最终答案或 max_steps
```

其中：

- `Application` 只负责装配。
- `RunCoordinator` 负责 run 生命周期和报告落盘。
- `AgentLoop` 负责 LLM-Tool 状态机。
- `ToolRegistry` 只负责工具注册和路由，不写 memory/telemetry。
- `Context` 只保存结构化 frame，不直接读取 MemoryStore。
- `PromptBuilder` 是唯一最终消息工厂。

## 上下文管理

每次请求模型时，消息按固定顺序构建：

```text
system: 固定规训和工具使用边界
user: workspace / memory / handoff / file summaries
assistant/user/tool: active history
user: current task
```

`Context` 内部保存两条历史：

- `_raw_frames`：完整原始历史，不因压缩被破坏。
- `_active_frames`：当前送入模型的历史子集。

当估算输入超过阈值时，`ContextCompactionHook` 会在 `pre_llm` 阶段：

1. 保护最近若干轮消息。
2. 将旧消息归档到 `.coding-agent/runs/<run_id>/dialog/*.jsonl`。
3. 调用压缩模型生成摘要。
4. 把摘要作为 assistant 消息放回对话流。

摘要形态类似：

```text
[CONTEXT COMPACTION] Earlier turns were compacted.

Archive: dialog/2026-05-22-xxxx.jsonl

## 目标
...

## 进度
...

## 下一步
...
```

工具结果的长输出不由压缩 Hook 处理，而是在 `ToolExecutor` 生成 tool message 时立即 offload 到：

```text
.coding-agent/runs/<run_id>/tool_result/*.txt
```

## Memory 与 Session

`session.py` 保存可恢复聊天记录：

```text
.coding-agent/sessions/*.json
```

这些记录用于 `chat --resume`，不会包含生成的 workspace context 或 current task wrapper。

`memory/` 保存长期运行状态：

```text
.coding-agent/memory/
  scratchpad.json
  handoff.md
  tool_index.jsonl
  file_summaries.json
```

当前 memory 主要记录工具事实、文件摘要、已知问题和 handoff。完整对话仍主要通过 session 保存；后续可以继续扩展 `session_search`、decision log 和 session memory 投影。

## 配置

运行初始化：

```bash
coding-agent init
```

示例配置：

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

设置 API key：

```bash
export OPENAI_API_KEY="..."
```

PowerShell：

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

常用参数：

- `path = "."`
- `max_entries = 200`
- `max_depth`

### `read_file`

读取 UTF-8 文件。

支持：

- `path`
- `start_line`
- `end_line`
- `max_chars = 50000`

大文件会自动截断，并在 metadata 里返回 `next_start_line`，提示下一次从哪里继续读。

### `write_file`

写入 UTF-8 文件，可自动创建父目录。

### `search_text`

搜索文本或正则。优先使用 `rg`，不可用时回退到 Python 遍历。

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

```bash
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.tmp-pytest
```

## 当前限制

- memory 里已有 handoff 读取链路，但压缩摘要写回 handoff 还可以继续加强。
- 完整历史搜索工具 `session_search` 尚未实现。
- 决策日志和用户偏好提取尚未实现。
- token 预算目前使用近似字符估算，不是模型 tokenizer。
- 当前 CLI 仍是简单命令行界面，没有全屏 TUI。
