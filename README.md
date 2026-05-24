# coding-agent

`coding-agent` 是一个面向本地代码仓库的 AI 编程助手 CLI。它通过 OpenAI-compatible `/chat/completions` 调用模型，把文件读写、文本搜索、补丁应用、命令执行和历史记忆搜索包装成受控工具，并用工作区沙箱、命令策略、上下文压缩和会话归档控制长任务复杂度。

完整架构说明见：[docs/architecture.md](docs/architecture.md)。

## 当前能力

- `coding-agent run "..."`：执行一次任务。
- `coding-agent chat`：交互式会话。
- `coding-agent chat --resume PATH`：恢复 conversation 目录或 session 目录。
- 原生 Function Calling：工具 schema 通过 API `tools` 数组传递，不在 system prompt 里重复列工具定义。
- 内置工具：`list_files`、`read_file`、`write_file`、`search_text`、`apply_patch`、`run_shell`、`session_search`。
- `read_file` 支持 `start_line` / `end_line`，默认 50KB 截断，并返回续读行号。
- `search_text` 和 `session_search` 优先使用 `rg`，不可用时回退 Python 搜索。
- phase/order 生命周期 Hook：按阶段编排上下文压缩、checkpoint、记忆投影和长输出 offload。
- pre-LLM 上下文压缩：旧消息归档，新 session epoch 从 compact summary 继续。
- conversation 级 checkpoint：写文件前确认 workspace、git 分支、HEAD 和目标文件 last-seen 状态，防止外部编辑被覆盖。
- 长工具输出通过 `ToolResultOffloadHook` 转存到 `tool_result/*.txt`。
- session memory 只保留 `handoff.md` 和 `scratchpad.json`，不保存 file summary cache 或 tool index。

## 存储结构

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

核心关系：

- `conversation`：一条长期任务线。
- `checkpoint`：一条 conversation 的工作区安全锁，压缩切 session 后仍保留。
- `conversation/memory`：跨 session 的长期记忆候选，当前由压缩时顺带提取。
- `session`：一个上下文窗口 epoch。触发压缩后会创建新 session。
- `run`：一次 AgentLoop 执行，挂在当前 session 下。

旧的 flat session 路径 `.coding-agent/sessions/*.json` 不再支持 `--resume`。

## 运行流程

```text
CLI
  -> Application 创建/恢复 conversation session
  -> RunStore 写入 active session/runs/<run_id>
  -> MemoryStore 写入 active session/memory
  -> CheckpointStore 刷新 conversation/checkpoints/checkpoint.json
  -> AgentLoop 构建 Context
  -> pre_llm hooks 必要时压缩并 rotate session
  -> PromptBuilder 生成 messages
  -> LLM 返回 assistant 消息或 tool_calls
  -> ToolExecutor 触发 pre_tool hooks，执行工具
  -> after_tool pipeline 投影 scratchpad 并 offload 长结果
  -> RunCoordinator 保存 active session messages
```

## 上下文压缩

触发条件：

```text
estimate_active_tokens > max_input_tokens * compact_threshold_ratio
```

压缩后：

1. 旧 session 保留完整 `session.json`。
2. 被移出上下文的旧消息写入当前 run 的 `dialog/*.jsonl`。
3. 压缩模型输出 JSON：`handoff` 和长期 `memories`。
4. 旧 session 标记为 `compacted`。
5. 新 session 创建为 `active`。
6. 新 session 的 `session.json` 只包含 assistant compact summary 和 protected recent messages。
7. 新 session 的 `memory/handoff.md` 写入 compact summary。
8. 新 session 的 `memory/scratchpad.json` 从旧 session 完整复制。
9. 长期 `memories` 追加写入 `conversation/memory/raw/YYYY-MM-DD.jsonl`。
10. `conversation.json.active_session_id` 最后原子切换到新 session。

这样压缩前后的完整消息不会混在同一个 session 里。

## 长期记忆

长期记忆是 conversation 级候选经验，不属于 session memory，也不会默认进入 prompt。

当前只在上下文压缩时提取，不额外增加一轮 LLM 调用。压缩模型同时输出：

```json
{
  "handoff": "给下一个 session 继续工作的 Markdown 摘要",
  "memories": [
    {
      "type": "procedural",
      "content": "修改工具 schema 后需要同步 tests/test_tools.py。",
      "confidence": 0.8,
      "reason": "这是项目内可复用的测试维护经验。"
    }
  ]
}
```

记忆类型：

- `personal`：用户稳定偏好、工作习惯、交流偏好。
- `procedural`：可复用做事方法、踩坑教训、测试/验证流程。
- `knowledge`：项目稳定事实、架构决策、目录结构、技术约束。

写入格式：

```text
.coding-agent/conversations/<conversation_id>/memory/raw/YYYY-MM-DD.jsonl
```

如果压缩模型没有输出合法 JSON，系统会把响应 fallback 为 handoff，`memories=[]`，不会阻塞压缩。

## session_search

`session_search` 是只读回溯工具，默认搜索当前 conversation 下所有 sessions。

参数：

- `query`
- `case_sensitive = false`
- `regex = false`
- `glob`
- `max_matches = 50`
- `scope = "current_conversation"`，可选 `current_session`
- `sources`: `memory` / `sessions` / `runs` / `dialog` / `tool_result`

典型用法：

1. 模型发现当前上下文缺少旧决策。
2. 调用 `session_search(query="关键字")` 定位归档路径。
3. 用 `read_file(path=..., start_line=...)` 读取原文。

## Checkpoint

Checkpoint 是系统安全机制，不是模型记忆，也不会进入 prompt。

它记录：

- workspace identity：workspace root、git branch、HEAD、git status、fingerprint。
- file records：模型通过 `read_file` 见过的文件状态，按路径替换更新。

写入规则：

- `write_file` 修改已存在文件前，必须确认该文件已被读过且内容没有漂移。
- `apply_patch` 的 update/delete/move source 必须已读且未漂移。
- `apply_patch` 的 add/move target 不能已经存在。
- 删除文件会留下 `exists=false` tombstone，避免记录丢失。

如果用户在 IDE 里改了文件，Agent 基于旧内容发起写入时，工具不会执行，而是返回结构化错误：

```json
{
  "ok": false,
  "code": "file_drift_detected",
  "path": "src/auth.py",
  "instruction": "Re-read this file with read_file before modifying it again."
}
```

模型收到这个 tool error 后应重新 `read_file`，再基于最新内容生成修改。

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
allow = ["python -m pytest", "pytest", "git status", "git diff", "git log"]
deny = ["rm", "del", "rmdir", "git reset", "git checkout", "powershell Remove-Item"]

[context]
max_input_tokens = 24000
compact_threshold_ratio = 0.8
compact_tail_ratio = 0.2
protected_recent_turns = 4
protected_tool_results = 6
handoff_max_chars = 6000
scratchpad_max_chars = 4000
show_cache_stats = true
```

PowerShell 设置 API key：

```powershell
$env:OPENAI_API_KEY = "..."
```

## 开发

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.tmp-pytest
```

## 当前限制

- token 预算仍是近似字符估算，不是模型 tokenizer。
- `session_search` 是文本/正则搜索，还没有语义向量检索。
- 压缩摘要质量取决于当前配置的模型。
