# Agent ReAct Loop 上线步骤与操作流程（2026-08-19）

> 分支：`feature/agent-react-loop` → `main`
> 上线内容：通用 agent 聊天路径（`query_stream`）切换到自研 ReAct 循环，替代旧单轮 `process()` 路径。

---

## 1. 本次变更概述

### 1.1 核心改动

- 新增 `sources/agents/react_loop.py`：框架无关的自研 ReAct 循环。一轮 = 一次绑定工具的 LLM 调用；LLM 不返回 `tool_calls` 即输出最终答案结束循环；工具执行结果以 `tool` 消息反馈，最多 `REACT_MAX_ROUNDS`（默认 10）轮。
- `GeneralAgent.create_agent()`（`sources/agents/general_agent.py:1569`）成为聊天主路径，由 `api_routes/core.py:1107` 调用；返回 `{'intent': 'long_task'}` 时 `core.py:1122` 转入长任务管道（与旧行为一致）。
- 工具集由 `build_tool_set`（`sources/agents/react_tools.py:221`）构建：
  - 固定工具 2 个：`fetch_patent_spec`（专利说明书解析）、`search_my_knowledge`（知识库检索，命中的工具通过 `mount_tools` 动态追加到本轮工具列表）
  - 知识库向量召回 TOP_N（默认 5，`REACT_TOOL_TOP_N`）个 type=1 知识工具 + 全部 type=3 长任务工具
- 检索增强（并行执行，`general_agent.py:1687`）：
  - 查询改写/检索阶梯 `build_search_queries`（`sources/long_task/search_query_builder.py`）
  - CPC 语义匹配 `match_query_to_cpc`（`REACT_CPC_EXPANSION` 开关）
  - 架构级技术解读 `interpret_query`（`REACT_INTERPRET_ENABLED` 开关，默认强模型 `openai/gpt-5.6-terra`）
  - `query_mode` 分类（`sources/long_task/query_mode.py`）：结构化查询（identifier/assignee/keyword/分析类）跳过 CPC 匹配与技术解读
- 循环内兜底：同一工具连续失败 2 次 → fallback 总结轮；达到最大轮数 → fallback；工具返回 `final` 标志 → 追加一次无工具轮直接出答案（`react_loop.py:168`）。
- 适配器：`make_llm_call`（`react_loop.py:205`，全程流式，`provider._get_langchain_llm(streaming=True)` + `bind_tools`）、`make_action_executor`（`react_tools.py`）、`make_event_emitter`（`react_loop.py:278`）。

### 1.2 不变量（保持不变）

- 长任务入口与管道不变：type=3 知识命中 → 返回 intent 标记 → `core.py` 长任务管线（分类 → 输入组装 → Celery 任务）。
- agent 池复用不变（`core.py`：最大 10 个、5 分钟空闲 TTL）；`create_agent` 每次请求重置全部 per-request 状态（`general_agent.py:1576`）。
- 多轮对话记忆机制不变（`_conversation_turns` + 系统提示中拼接 Previous conversation 块）。

---

## 2. 配置文件（config.ini）

无新增配置段，沿用 `[MAIN]` 的 provider/model：

```ini
[MAIN]
provider_name = openrouter          ; 生产建议固定，勿用多 provider 轮换
provider_model = openai/gpt-5.6-terra
```

- 注意：`make_llm_call` 强制 `streaming=True`（MiniMax 非流式返回空内容，见 `llm_provider.py:750` 注释）；全链路 `temperature=0`。

---

## 3. 环境变量（.env）

### 3.1 循环与工具（必读）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `REACT_MAX_ROUNDS` | 10 | ReAct 循环最大轮数（`react_loop.py:17`） |
| `REACT_TOOL_TOP_N` | 5 | 知识库召回绑定的知识工具数（`react_tools.py:49`） |
| `REACT_RELEVANT_TOP_N` | 10 | 相关度排序结果在最终答案中列出的条数（`general_agent.py:61`） |

### 3.2 检索增强开关与阈值

| 变量 | 默认值 | 说明 |
|---|---|---|
| `REACT_QUERY_MODE_ENABLED` | 1 | query_mode 分类开关（`query_mode.py`） |
| `REACT_CPC_EXPANSION` | 0 | CPC 语义匹配与召回开关（开启前必须先构建 `data/cpc` 产物，见 5.2） |
| `REACT_INTERPRET_ENABLED` | 1 | 架构级技术解读开关（`technical_interpretation.py`） |
| `REACT_INTERPRET_MODEL` | openai/gpt-5.6-terra | 解读用模型（强模型，单独计费注意） |
| `REACT_LADDER_MAX_HITS` | 300 | 命中数超限截断（回退 `REACT_TIGHTEN_SUGGEST_THRESHOLD`） |
| `REACT_LOW_HIT_FEEDBACK_THRESHOLD` | 10 | 低命中反馈触发阈值（`react_tools.py:50`） |
| `REACT_RELEVANCE_RANK` | 1 | 搜索结果相关度排序开关 |
| `REACT_USPTO_SORT_FIELD` | _score | USPTO 排序字段 |
| `CPC_DATA_DIR` | data/cpc | CPC 向量库目录（`cpc_semantic.py`） |
| `CPC_VECTOR_LEVEL` | （空） | CPC 向量匹配档位 |

### 3.3 接地合成与 recall 扩展（专利检索场景）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `REACT_GROUNDED_MIN` | 45 | 接地合成触发最低评分（`recall_sources.py`） |
| `REACT_GROUNDED_POOL_MIN` | 120 | 接地合成触发最小池容量 |
| `REACT_GROUNDED_HEAD` | 30 | 接地合成取池头条数 |
| `REACT_FAMILY_SCORE` | 1 | 同族补分开关 |
| `REACT_FAMILY_SCORE_BUDGET` | 30 | 同族补分预算 |
| `REACT_FAMILY_SEED_MIN` | 4 | 同族种子最小规模 |
| `REACT_RECALL_MAX_FAMILY_NUMBERS` | 12 | recall 同族号上限 |
| `REACT_RECALL_NUMBER_BATCH` | 20 | 申请号批处理大小 |
| `REACT_RECALL_MAX_CPC` | 3 | recall 扩展最大 CPC 数 |
| `REACT_RECALL_CPC_PER_CODE` | 50 | 每个 CPC 召回条数 |
| `REACT_RECALL_CPC_TOP_PER_CODE` | 300 | 每个 CPC 排序窗口（`recall_sources.py:50`） |
| `REACT_AUTO_FEEDBACK_MAX_QUERIES` | 2 | 低命中自动反馈查询数 |
| `REACT_AUTO_LADDER_MAX_QUERIES` | 6 | 自动阶梯查询数 |
| `REACT_AUTO_ROUND_MAX_QUERIES` | 2 | 系统驱动第二轮查询数 |
| `REACT_MISSING_DIR_MIN_CANDIDATES` | 3 | 缺失方向反馈最小候选数 |
| `REACT_MISSING_DIR_MIN_SCORE` | 4 | 缺失方向反馈最小评分 |
| `REACT_POOL_MAX_PAGES` | 2 | 候选池最大抓取页数 |
| `REACT_POOL_MAX_TOTAL_PAGES` | 1000 | 候选池总页数上限 |
| `REACT_MAX_PATENT_LIST_ITEMS` | 100 | 专利列表工具输出条数上限（`react_tools.py:104`） |
| `REACT_SCORE_PROVIDER_FAMILY` | — | 评分模型提供商家族 |
| `REACT_SCORE_MODEL` | — | 评分模型名 |

### 3.4 工具结果裁剪（大列表/长文本）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `GENERAL_AGENT_MAX_ITEM_CHARS` | 15000 | 单条工具结果最大字符（`general_agent.py:57`） |
| `GENERAL_AGENT_MAX_VALUE_CHARS` | 10000 | 单字段最大字符 |
| `GENERAL_AGENT_SMALL_LIST_THRESHOLD` | 3 | 小列表判定阈值 |
| `GENERAL_AGENT_SUMMARY_MAX_CHARS` | 120000 | 大列表摘要预算（`general_agent.py:310`） |

### 3.5 基础设施

| 变量 | 说明 |
|---|---|
| `USPTO_API_KEY` | USPTO 专利检索必需 |
| `MCP_FINDER_API_KEY` | MCP Finder 工具开关（缺失时 agent 禁用） |
| `OPENROUTER_API_KEY` | 当前 provider 密钥（按 config.ini 的 provider 对应更换） |
| `REDIS_HOST` / `MYSQL_HOST` | 沿用现有 Redis/MySQL 配置，无变化 |

---

## 4. 数据库变更

**无 schema 变更。** `long_tasks` 表沿用旧结构。

知识库内容提醒（影响检索质量，非阻塞）：

```sql
SELECT k.id, k.question, k.description, k.`type`, t.title
FROM knowledge k LEFT JOIN tools t ON k.tool_id = t.id AND t.status = 1
WHERE k.status = 1 AND k.scene_id = 1 AND k.`type` IN (1, 3);
```

- type=3 长任务卡片的 `description` 会原样注入工具描述（`_long_task_description`），确保写清"该工具能做什么、何时用"。
- type=1 知识卡片的 `answer` 作为 usage guide 注入工具描述，**截断至 1000 字符、描述总长 1600**（`general_agent.py:1429-1437`）——重要参数说明（如 path 占位符语义）务必放在 answer 开头。

---

## 5. 部署执行步骤

### 5.1 环境变量先行（部署前必做）

将第 3 节的 `REACT_*` 开关按生产预期写入服务器 `.env`（至少确认）：

```bash
# .env（服务器）
REACT_QUERY_MODE_ENABLED=1
REACT_INTERPRET_ENABLED=1
REACT_INTERPRET_MODEL=openai/gpt-5.6-terra
REACT_CPC_EXPANSION=1          # 开启 CPC 匹配与召回；前置条件见 5.2
REACT_MAX_ROUNDS=10
REACT_TOOL_TOP_N=5
REACT_RELEVANT_TOP_N=10
```

未设的变量全部有代码内默认值（见第 3 节表），不设也能跑，但建议显式确认。

### 5.2 数据与脚本准备（data/cpc + scripts）

CPC 语义匹配与召回依赖 `data/cpc/` 下的两类**构建产物**，它们**不在 git 仓库里**（git 只跟踪原始素材），必须在服务器上构建：

| 产物 | 构建脚本 | 用途 | 缺失时的行为 |
|---|---|---|---|
| `data/cpc/cpc_index.db` | `scripts/build_cpc_index.py` | `fetch_by_cpc` 本地解析 CPC→专利号（`recall_sources.py:180`） | 优雅降级返回 `[]`，召回扩展不生效，不报错 |
| `data/cpc/cpc_title_vectors*.npy` | `scripts/build_cpc_vectors.py` | `match_query_to_cpc` 语义匹配向量缓存（`cpc_semantic.py:119`） | 加载返回 None，匹配跳过，不报错 |

```bash
# 已在 git 中的素材（无需下载）：
#   data/cpc/CPCSchemeXML202608.zip
#   data/cpc/cpc_titles_main_groups.json / cpc_titles_subgroups.json

# ① 构建 CPC→专利号索引（一次性 + 每月随 USPTO MCF 更新刷新）
#    需 .env 中 USPTO_API_KEY（经 ODP datasets API 下载 MCF 月度包）
python scripts/build_cpc_index.py

# ② 构建 CPC 标题向量缓存（一次性；子组档默认，~15 万 × 1024 维）
#    需 .env 中 EMBEDDING_PROVIDER / EMBEDDING_API_KEY / EMBEDDING_BASE_URL
python scripts/build_cpc_vectors.py --groups sub     # 子组档（CPC v2，默认）
# python scripts/build_cpc_vectors.py --groups main  # 主组档
```

- 档位控制：`CPC_VECTOR_LEVEL`（空=主组，`sub`=子组），与 `--groups` 保持一致。
- 子组档约 15 万条标题，向量构建耗时与内存占用较大，建议低峰期执行、预留内存；构建中途失败可重跑（增量覆盖）。
- 可选诊断脚本（本分支新增，非部署必需，用于验证/排查）：
  ```bash
  python scripts/uspto_cpc_probe.py     # USPTO CPC 检索探针
  python scripts/recall_probe.py        # recall 扩展（同族/CPC）探针
  python scripts/interpret_probe.py     # 技术解读探针
  python scripts/mcf_probe.py           # MCF 数据解析探针
  ```
- 上线后确认产物就位：
  ```bash
  ls -la data/cpc/    # 应有 cpc_index.db 与 cpc_title_vectors*.npy
  ```

### 5.3 后端（Python）

本分支**未改动 `requirements.txt` / `Dockerfile.backend`**，无需 pip install 或重建镜像。代码经 volume 挂载（`./:/app`），重启即生效：

```bash
# 拉取代码
git pull --ff-only

# 重启后端容器
docker compose up -d backend          # 容器存在时等效 restart
# 或
docker compose restart backend

# 若生产另有 celery worker 进程/容器，同样需要重启
```

### 5.4 前端（frontend/nextjs，Cloudflare Pages）

本分支改动了 nextjs 前端（聊天落地页、`useChatStream.ts` SSE 事件处理、`package.json` 新增 devDependency `puppeteer-core`），需重新构建并发布：

```bash
cd frontend/nextjs
npm install                              # lockfile 已变更
npm run pages:build                      # npx @cloudflare/next-on-pages → .vercel/output/static
wrangler pages deploy .vercel/output/static
# wrangler.toml 已配置 pages_build_output_dir，也可直接: wrangler pages deploy
```

### 5.5 服务验证

```bash
# 后端健康
curl http://localhost:7777/health        # 或线上域名 /health

# 启动无报错
docker compose logs backend --tail 50 | grep -i "error\|exception"

# 前端线上可访问（落地页 + 聊天页正常渲染）
```

---

## 6. 前端 SSE 事件契约（变化点）

| 事件 | 触发点 | 前端处理 |
|---|---|---|
| `status` | 每步工具调用前 + 各阶段（构造检索式/分析问题） | transient，分析中状态条（行为不变） |
| `step` | 每步工具调用（`react_loop.py:101`） | 步骤时间线：round/thought/action/params_brief/reasoning_text |
| `observation` | 每步工具返回（`react_loop.py:133`） | 步骤时间线：result_brief |
| `agent_elapsed` | 循环结束（`react_loop.py:198`） | elapsed_seconds + steps；计时起点 = 请求开始（含 create_agent 阶段） |
| `patent_ids` | 对话产物生成后（`general_agent.py:138`） | 前端隐藏存储，供 conversation_refs 追问携带 |
| `token` / `artifact_*` | 不变 | 不变 |

注意：**工具轮没有正文 token**（OpenAI 兼容 API 只返回 tool_calls），工具期间的进度只能依赖 `status`/`step` 事件，token 开始即应清空状态列表（`frontend` 已实现）。

---

## 7. 上线验证清单

### 7.1 自动化测试（部署前必须全绿）

```bash
pytest tests/test_react_loop.py \
       tests/test_react_tools.py \
       tests/test_react_integration_general_agent.py \
       tests/test_query_mode.py \
       tests/test_technical_interpretation.py \
       tests/test_search_query_builder.py \
       tests/test_recall_sources.py \
       tests/test_relevance_gate.py \
       tests/test_sse_callback_events.py \
       tests/test_general_agent_prune_summary.py
```

### 7.2 手工验证场景

1. **语义技术搜索**（如"动力电池热失控防护"类提问）：观察步骤时间线中阶梯调整（0 结果 → 载体词重试 → 放宽）；最终答案按相关度列出 top-N 项并附一句适配理由。
2. **结构化查询**（assignee / 申请号 / 关键词）：`query_mode = structured`，跳过 CPC/解读（日志 `search_rewrite — mode=structured`）。
3. **长任务入口**：触发 type=3 工具（如专利布局分析）→ 循环返回 intent → 转长任务管道，进度事件正常。
4. **多轮追问**：第一轮产物后收到 `patent_ids` 事件；追问"里面第 3 件专利的说明书"能命中 conversation_refs。
5. **兜底路径**：构造失败场景验证 fallback（连续失败 2 次 / 超 10 轮）输出中文总结而非报错。
6. **语言跟随**：中英文提问各自全程对应语言回复。

### 7.3 监控

- 日志：`general_agent.log`（搜索改写、interpretation、接地触发、recall 均有关键日志）。
- 观察指标：`agent_elapsed` 的 elapsed_seconds 分布、轮数分布（`steps`）——异常放大需排查阶梯/召回循环。

---

## 8. 回滚方案

1. **代码回滚**：`git revert` 或切回上一发布提交。关键调用点仅一处：`api_routes/core.py:1107` 的 `general_agent.create_agent(...)`；旧 `process()` 路径仍保留在 `general_agent.py` 中未删除，如需恢复旧行为需还原 `core.py` 的 `generate()` 实现。
2. **配置回滚**：`.env` 中恢复旧开关（如 `REACT_QUERY_MODE_ENABLED=0` 可单独关闭新检索增强，不关闭循环本身）。
3. **数据安全**：无迁移、无写库 schema 变更，回滚不涉及数据修复。

---

## 9. 注意事项与已知限制

- **MiniMax 必须流式**：`make_llm_call` 强制 `astream`，非流式返回空内容（`llm_provider.py:750`）。
- **OpenAI 兼容 API 的 tool 消息约束**：assistant 的每个 `tool_call` 必须有对应 `tool` 输出，否则 400（"No tool output found"）；fallback 前已补全本轮剩余 tool 消息（`react_loop.py:150-164`）。
- **agent 池复用**：`create_agent` 必须重置 per-request 状态（`general_agent.py:1576-1603` 的状态重置清单），新增状态字段时必须同步重置，否则跨请求串扰。
- **工具描述注入**：知识卡片 answer 截断至 1000 字符，关键参数说明放开头。
- **大列表输出**：超过预算的工具结果被摘要化或截断（第 3.4 节），最终展示由 `_stream_raw_items` 逐条流式输出——前端需支持中途插入。
