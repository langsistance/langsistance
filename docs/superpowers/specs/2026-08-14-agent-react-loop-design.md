# Agent ReAct 循环升级设计

- **日期**: 2026-08-14
- **状态**: 已确认（三节设计均经用户确认）
- **范围**: `langsistance/` 后端聊天管线 + `frontend/nextjs/` 聊天窗口渲染

## 1. 背景与目标

现状：查询 → `select_knowledge_tool_with_llm` 从候选里**只选一个** knowledge+tool →
`openai_create`（LangChain `create_agent`）单发调用 → 格式化输出。agent 没有
「思考→决策→行动→观察」循环，无法自主串联多个知识/工具。

目标：升级为轻量 ReAct 循环的通用 agent，对话交互对齐 workbuddy 风格：

1. agent 在循环中不断思考、决策、行动、观察，自主决定用哪些 knowledge/工具、按什么顺序拼接
2. 行动面 = 用户全部可用 knowledge + 对应工具 + 长任务（type=3）
3. 执行中持续输出**简洁**状态与思考行（用户不觉得卡死）；最终回答产出时
   折叠隐藏全部过程，只留「已用时间 · N 步」+ 最终回答；点击展开可见
   结构化步骤时间线（每轮：思考 / 工具+参数摘要 / 观察摘要）
4. 最终产物是专利列表 → 复用现有结果列表窗口展示；其他交互（结果页、
   下载、prosecution、长任务卡片）不变

## 2. 非目标（不改）

- 结果页分栏视图、artifact 下载、prosecution 就地分析 —— 全部保持现状
- Celery 长任务后端、knowledge CRUD、工具执行机制（`execute_backend_tool_request`、
  `outbound_http`、`get_dynamic_tools` 的 push=1/2/3 语义）—— 复用不改
- CLI 路径（`GeneralAgent.process`）、`select_knowledge_tool_with_llm` 本身 —— 保留
- 不做 DB 迁移、不删除 `WorkflowExecutor`（休眠，见 §4.4）

## 3. 架构：ReAct 循环

新模块 `sources/agents/react_loop.py`（目标 150~250 行核心循环），`GeneralAgent`
聊天管线的 `create_agent`/`invoke_agent` 后半段改造为构建工具集 + 启动循环。

### 3.1 控制流

```
入口（改造后的 create_agent）:
  1. 语言检测 → 状态 "正在分析您的问题..."
  2. 构建工具集（§4）
  3. 组装消息: 固定系统前缀（格式/语言规则）+ 多轮历史 + 用户问题
  4. 启动 react_loop，每轮事件经 SSECallbackHandler 发出

循环（round 1 .. MAX_ROUNDS）:
  ├─ LLM 调用（流式，bind_tools(当前工具列表)）
  │    ├─ 产出 tool_calls → 行动
  │    └─ 产出最终回答 → 发 agent_elapsed → 结束循环
  ├─ 行动: 发 step 事件（思考行）
  │    ├─ knowledge 工具 → 执行（§5.1）
  │    ├─ search_my_knowledge → 检索，下一轮动态挂载（§4.1）
  │    └─ 长任务工具 → 提交 Celery → 输出说明 → 结束（§5.3）
  ├─ 观察: 发 observation 事件（结果摘要），ToolMessage 压回历史 → 下一轮
  └─ 保护: 每轮间检查 request_stop；轮次上限兜底（§7）
```

**约束与决策：**

- 每轮 LLM 调用前重新 `llm.bind_tools(当前工具列表)` —— 检索后动态挂载工具
  由此实现，手写循环天然支持
- **只有最终轮（无 tool_calls）的 token 流式转发给前端**作为回答内容；
  中间轮（带 tool_calls）的 token 不转发（这些轮通常无内容文本），其推理
  文本（若模型返回）存入该轮时间线的思考字段
- 中间轮也走流式调用（MiniMax 非流式返回空内容的历史问题），由循环层抑制转发
- `_ResponseCollector` 继续包裹 handler，最终回答文本照旧收集用于多轮存储
  （`_store_current_turn`）
- 思考行由模板生成（「第 N 步 · 正在调用「工具名」」+ 关键参数），保证简洁；
  时间线存完整思考/参数摘要/观察摘要

### 3.2 常量（可 env 覆盖）

| 常量 | 默认 | env |
|---|---|---|
| 最大轮数 | 10 | `REACT_MAX_ROUNDS` |
| 预注册知识数 top-N | 5 | `REACT_TOOL_TOP_N` |
| 专利列表截断条数 | 100 | `REACT_MAX_PATENT_LIST_ITEMS` |

## 4. 工具供给（方式 C）

### 4.1 `search_my_knowledge` 元工具

- 内部工具，参数 `{query: str}`
- 执行：复用 `get_knowledge_tool_candidates` 向量召回 + type=3 全量候选
- 返回给 LLM：top 匹配的 `{knowledge_id, question, 类型, 工具简介}`（精简，
  防 prompt 膨胀）；**排除 type=2**
- 下一轮循环把这些匹配工具 bind 进去

### 4.2 知识工具（预注册）

入口时向量召回 top-5 knowledge（复用 `get_knowledge_tool_candidates`），经现有
`get_dynamic_tools` 逻辑注册为 dynamic tool。push=1/2/3 机制与参数模板照旧。

### 4.3 长任务工具（type=3）

每个 type=3 knowledge 注册为一个工具（名称=清理后的标题）。被调用时：

1. 循环输出「已创建批量分析任务」说明文本（语言随系统）
2. 循环发出与今天相同的 `{'intent': 'long_task', knowledge, tool_info}` 标记并终止
3. `run_pipeline` 现有分支照旧处理：场景分类 → Celery 提交 → `long_task_created`
   → 前端进度卡片

### 4.4 workflow（type=2）退役

- type=2 的 workflow 知识与工具**不再进入循环供给**（从候选与预注册中排除）
- 循环中 agent 自主决定如何拼接 type=1 知识/工具，取代 workflow 编排
- `create_agent` 中的 workflow 特殊分支移除；`WorkflowExecutor` 代码保留不删（休眠）
- 存量 type=2 数据不迁移、不删除，自然休眠

## 5. 行动处理器

### 5.1 knowledge 工具执行

- 复用 `execute_backend_tool_request` / 动态工具机制
- 搜索结果 `raw_items` 照旧进 `_pending_raw_items` → `invoke_agent` 现有
  流式批处理 → artifact → 结果页联动，**全部保持现状**

### 5.2 结果条数规则

判定依据：复用 `_infer_result_source`（按工具 URL）：

- **专利搜索列表** → `_pending_raw_items` 生成处截断为前 100 条；observation
  摘要注明「共 N 条，展示前 100 条」；artifact/结果页只收这 100 条
- **专利文档列表**（`uspto_documents` 类）→ 不限制，全部进入

### 5.3 长任务执行

见 §4.3。提交失败 → 复用现有分支错误提示，循环结束。

## 6. SSE 事件协议（`sse_callback.py`）

新增 3 类，其余全部不变：

```jsonc
{"type": "step",         "round": 2, "thought": "第 2 步 · 正在调用「美国专利检索」",
                         "action": "uspto_search", "params_brief": "query=..."}
{"type": "observation",  "round": 2, "result_brief": "返回 120 条，截断为前 100 条"}
{"type": "agent_elapsed", "elapsed_seconds": 3.2, "steps": 5}
```

- 保留 `token / status / artifact_* / patent_ids / long_task_created / error / end` 语义
- 时间线由前端从 step/observation 自行累积；`agent_elapsed` 在循环结束时发出
  （最终回答 token 流完后、`end` 事件之前），触发前端折叠为「已用时间 · N 步」
- 后端时间打点：循环入口 → 结束时计算 elapsed（含工具执行时间）
- 现有 `on_tool_start/on_tool_end` 死代码不启用、不删除

## 7. 错误处理

| 情形 | 行为 |
|---|---|
| 工具执行异常 | observation 记错误摘要 → 循环继续（LLM 可换工具）；同一工具连续 2 次失败 → 终止循环，基于已有观察给兜底回答 |
| LLM 调用异常 | 现有 `error` SSE 事件，前端提示重试 |
| 轮次达上限（10） | 基于已有观察让 LLM 生成兜底回答，正常结束 |
| 无匹配 knowledge | agent 直接回答，提示可去社区找共享知识（现状文案保留） |
| 用户中断 | 每轮之间检查 `request_stop`，立即收尾 |
| 长任务提交失败 | 复用现有分支错误提示，循环结束 |

## 8. 前端交互（workbuddy 风格）

只改聊天消息渲染（`MarkdownMessage` + `useChatStream` + `ChatContext`），结果页不动。

```
流式执行中:
  ✓ 第1步 · 已查询「美国专利检索」(24ms)
  ⏳ 第2步 · 正在调用「专利文档下载」…        ← 最新一行 spinner，持续更新
  [状态: 正在整理结果...]                     ← 现有 transientStatus 机制

完成后（自动折叠）:
  ⏱ 已用时间 3.2 秒 · 5 步          ▾        ← 点击展开/收起
  ┌─────────────────────────────────────┐
  │ 第1步 思考: 需要检索美国专利          │
  │   调用: 美国专利检索 (query=...)      │
  │   观察: 返回 120 条，截断为前 100 条  │
  └─────────────────────────────────────┘
  [最终回答 Markdown 流式内容]
```

- `ChatMessage` 新增 `agentSteps?: AgentStep[]`、`elapsedSeconds?: number`
  - `AgentStep = {round, thought, action, paramsBrief, observationBrief,
    reasoningText?, status: 'running'|'done'|'error'}`
  - `thought` = 模板思考行（展开时间线也用它做标题）；`reasoningText` =
    模型返回的原始推理文本（若该轮有，展开时展示在思考行下方；无则省略）
- `useChatStream` 处理 `step`（追加/更新进行中行）、`observation`（同轮合并显示）、
  `agent_elapsed`（设置 elapsed，触发折叠）
- 执行中只显示**已完成步骤行 + 最新进行中一行**，不堆叠冗余内容
- i18n 新增中英文键（agentThinking、stepsCount、elapsedSeconds、展开/收起等）
- 思考行模板双语（后端按 `_detect_lang` 生成）

## 9. 测试

后端（`PYTHONUTF8=1 python -m pytest`）：

- 新 `tests/test_react_loop.py`（mock LLM + 假工具）：
  - 无工具调用 → 直接回答
  - 单工具一轮完成
  - 多轮串联：检索 → 挂载 → 调用 → 回答
  - 长任务工具调用 → 发出 intent 标记并终止
  - 专利列表 100 条截断；文档列表不截断
  - 轮次上限 → 兜底回答
  - 工具失败 → observation 错误摘要 → 换路继续；同工具连续 2 次失败 → 终止
- 现有相关套件（`test_patent_detail_api`、`test_uspto_download`、`test_text_extractor`、
  `test_http_outbound`、`test_general_agent_prune_summary`、`test_tool_result_filter`
  等）保持全绿

前端（`node --test lib/*.test.mjs` + `npm run build`）：

- `useChatStream` 对 step/observation/agent_elapsed 的累积与折叠状态单测
- 现有 122 个单测保持通过；生产构建通过

回归实测（test.copiioai.com）：搜索 → 结果页跳转、下载 artifact、prosecution
就地、长任务卡片、多轮跟进。

## 10. 实现文件清单

| 文件 | 动作 |
|---|---|
| `sources/agents/react_loop.py` | 新增：循环核心 |
| `sources/agents/general_agent.py` | 改：入口构建工具集 + 启动循环；移除 workflow 分支；保留 long_task intent 处理 |
| `sources/callback/sse_callback.py` | 改：新增 step/observation/agent_elapsed 事件方法 |
| `api_routes/core.py` | 基本不动（长任务 intent 分支、SSE 透传保持） |
| `tests/test_react_loop.py` | 新增 |
| `frontend/nextjs/contexts/ChatContext.tsx` | 改：ChatMessage 字段 + 状态 |
| `frontend/nextjs/lib/useChatStream.ts` | 改：新事件消费 |
| `frontend/nextjs/components/app/MarkdownMessage.tsx` | 改：步骤行 + 折叠时间线渲染 |
| `frontend/nextjs/lib/app-i18n/locales/en.ts + zh.ts` | 改：新键 |
| `frontend/nextjs/styles/app.css` | 改：步骤行/时间线样式 |

不动：`knowledge.py`、`workflow_executor.py`（休眠）、`sources/tools/*`、
Celery 长任务后端、结果页组件。
