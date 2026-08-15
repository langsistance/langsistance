# 聊天路径专利检索 + 自主下载分析 设计文档

**日期:** 2026-08-15
**范围:** ReAct 聊天管线（GeneralAgent）的专利检索类查询质量
**关联:** [2026-08-14-agent-react-loop-design.md](./2026-08-14-agent-react-loop-design.md)（ReAct 循环本体）、[2026-08-15-patent-search-quality-phase1.md](../plans/2026-08-15-patent-search-quality-phase1.md)（长任务管线 PHASE0 质量）

---

## 1. 背景与问题

2026-08-15 实测（查询「帮我查找工业中在线干燥空气源提供和设备内部环境湿度精准控制的相关专利」）暴露聊天路径三个缺陷：

1. **搜索观察只有计数**：工具返回 raw_items 时，循环观察仅一句「返回 N 条记录」——LLM 看不到任何真实条目，回答只能写「三组结果」之类的套话，无法筛选分析。
2. **检索词噪声**：LLM 直译 q 并混入 filler 词（"online dry air source supply industrial equipment internal environment humidity precise control **patent**"）→ USPTO 空格 OR 语义命中 8,814,019 条，top 结果全是无关领域（放射治疗、显示器面板）。
3. **无深挖能力**：用户追问某篇专利细节时，agent 没有下载说明书/权利要求的路径；说明书下载实现只存在于长任务管线（celery_worker 内部函数），聊天路径不可复用。

## 2. 产品方向（用户确认）

- 这类「查找相关专利」查询**不走批量长任务**——由聊天 agent 自行「检索 → 判断 → 分析 → 回答」。
- 成功画像：回答给出 5-10 篇切题专利（标题/申请号/申请人/状态 + 每篇一两句相关性分析）；用户追问某篇时，agent **自主判断**调用下载工具拿说明书/权利要求深挖。
- 下载与分析实现参考批量长任务管线（直接 USPTO API + SPEC 选择 + 文本提取）。

## 3. 关键决策（已与用户确认）

| # | 决策点 | 选择 |
|---|---|---|
| D1 | 成功画像 | 清单+简要分析；深挖由用户追问触发 |
| D2 | 搜索质量保证 | 确定性改写（复用 `search_query_builder`），不依赖提示词现场发挥 |
| D3 | 下载能力形态 | 循环内置工具 `fetch_patent_spec`，下载逻辑抽取共享模块，不依赖知识召回 |
| D4 | 全文上下文管理 | flash 提炼后进循环（下载全文 → 结构化提炼 → 提炼结果入观察） |
| D5 | 通用性 | 全部实现面向**通用专利场景**：任何模块不得为特定查询/技术领域硬编码关键词、阈值或分支；改写与提炼的输入均为用户原始问题，与领域无关；验收用多领域查询基准集 |

## 4. 架构

三个独立单元，全部落在 ReAct 聊天路径：

```
┌─ 用户查询 ─────────────────────────────────────────────┐
│ ReAct 循环                                              │
│  Round: LLM 决定调搜索工具                               │
│    → q 确定性改写（用户原始问题 → OR 组检索式）           │
│    → 观察 = 条目摘要（top-20 申请号|标题|申请人|日期|状态）│
│    → LLM 判断相关性 → 「清单+简要分析」回答               │
│  用户追问某篇 → LLM 调 fetch_patent_spec(专利号)         │
│    → 共享下载模块（长任务同款实现）                      │
│    → flash 提炼（发明点/技术问题/方案/权利要求要点）      │
│    → 观察 = 提炼结果 → LLM 深挖回答                      │
└────────────────────────────────────────────────────────┘
```

### 文件结构

| 文件 | 职责 |
|---|---|
| `sources/long_task/uspto_download.py` | **新建**。从 celery_worker.py 抽取的 USPTO 说明书下载实现（共享模块） |
| `sources/long_task/patent_distill.py` | **新建**。说明书全文 flash 提炼（纯函数格式化 + async 提炼入口） |
| `sources/agents/react_tools.py` | **修改**。`_items_digest`、`fetch_patent_spec` 工具注册与执行分支、确定性改写接入 |
| `celery_worker.py` | **修改**。删除被抽取的下载函数，改为从共享模块 import（行为不变） |
| `tests/test_uspto_download.py` | **新建**（共享模块纯逻辑） |
| `tests/test_patent_distill.py` | **新建** |
| `tests/test_react_tools.py` | **修改**（追加新测试） |

## 5. 单元 1：搜索观察内容增强

**现状**：`dynamic_backend_tool_function`（general_agent.py）对含 raw_items 的结果返回「返回 N 条记录」计数消息；完整列表走 `_pending_raw_items` 展示给用户，LLM 侧拿不到条目内容。

**设计**：
- 新增纯函数 `_items_digest(raw_items, limit=20, lang="zh") -> str`（react_tools.py）：
  - 复用 `sources/long_task/candidate_metadata.build_candidates` 扁平化 USPTO 条目，每行格式：`申请号 | 标题 | 申请人 | 申请日 | 状态`，cap 20 条，超出注明 `…共 N 条`
  - 非 USPTO 结构（build_candidates 产出为空）→ 兜底：raw JSON 截断（前 3000 字符）
- `make_action_executor` knowledge 分支：有 `raw_items` 时，观察文本 = 条目摘要 + 「完整列表已展示」提示；无 raw_items 时保持 `_summarize_observation`（300 字符）不变
- 用户侧完整列表展示（`_pending_raw_items` → 100 条截断 → 前端列表/Excel）**不变**
- 观察长度上限：搜索摘要 ~3000 字符；其余结果维持 300

## 6. 单元 2：确定性查询改写

**接入点**：`execute_action` 的 knowledge 分支，条件 = `tool_info.push == 2` 且 `is_keyword_search_tool(tool_info)`（Google 专利关键词搜索工具一并覆盖；assignee/文档类工具不动）。

**流程**：
1. 改写输入 = `agent._last_user_prompt`（用户原始问题，不用 LLM 已生成的 q）
2. 首次执行时调用 `build_search_queries(用户问题, agent 的 provider)`，结果缓存到 `agent._search_rewrite`（同轮多次搜索复用，只调一次）
3. 取 `queries[0]` 覆盖 args 的查询键——防御性处理三种形态：`args["q"]`、`args["query"]`、`args["params"]`（JSON 字符串内嵌 q/query 键）
4. 降级：改写失败/返回空 → args 原样（现状行为），仅记录日志

## 7. 单元 3：内置下载工具 + 共享模块 + flash 提炼

### 7.1 共享模块抽取

- 将 `celery_worker.py` 中以下实现**原样**搬入 `sources/long_task/uspto_download.py`：
  - `_download_uspto_patent_direct`（申请号规范化 → documents 列表 → SPEC 候选收集 → LLM 选择优先 SPEC → 多 SPEC 下载拼接 → XML/DOCX/PDF 文本提取 → binary fallback）
  - 其依赖的 `_uspto_get_with_retry` 与文本提取辅助段
- 公开签名：`async def download_uspto_patent_text(patent_id: str, spec_selector_provider=None, logger=None) -> tuple[str | None, bytes | None]`
- 模块级 `_pipeline_logger` 改为 logger 参数注入（celery_worker 传原有 logger；聊天路径传 general_agent logger）
- `celery_worker.py` 改为 import 共享模块，**行为逐字节不变**（回归套件守护）；长任务管线与聊天路径共用同一实现

### 7.2 内置工具 fetch_patent_spec

- 注册位置：`build_tool_set`，与 `search_my_knowledge` 并列**常驻**（不依赖知识召回）
- 参数 schema：`{"patent_id": "8 位申请号（可含 US 前缀/逗号/斜杠）"}`
- 工具描述引导：「当用户要求深入了解某篇专利的技术方案、权利要求或说明书内容时调用；输入专利申请号」
- 执行器（`make_action_executor` 新增 `kind == "patent_spec"` 分支）：
  1. 调 `download_uspto_patent_text(patent_id, spec_selector_provider=agent.llm, logger=agent.logger)`
  2. 得到文本 → `distill_patent_spec(text, user_query, provider)` 提炼
  3. 观察 = 提炼结果格式化文本（结构化、约 1-2k 字符）
  4. 下载失败 → 观察 = `Error: 说明书下载失败（原因）`（循环可换策略或如实告知用户）

### 7.3 flash 提炼（patent_distill.py）

- 入口：`async def distill_patent_spec(text: str, query: str, provider) -> dict`
- 提炼 prompt 输出 JSON：`{"发明点": ..., "解决的技术问题": ..., "技术方案": ..., "权利要求要点": ..., "与用户问题的相关性": ...}`
- 纯函数 `format_distilled(distilled: dict, lang="zh") -> str` 负责观察文本格式化
- 提炼失败降级：观察 = 全文前 16000 字符截断（与长任务 `patent_analyzer.py:174` 的 `patent_text[:16000]` 口径一致）
- 提炼用 agent 自己的 provider（`agent.llm.complete_json`）；MVP 不新增配置，后续可换 flash 模型

## 8. 错误处理与降级（总表）

| 环节 | 失败时行为 | 循环影响 |
|---|---|---|
| 改写 LLM 失败/空 | args 原样（现状） | 无 |
| 摘要生成异常 | 回退现有计数消息 | 无 |
| 下载失败（无 SPEC/网络/无文本） | 观察 = 明确错误文本 | LLM 可换策略或向用户说明 |
| 提炼 LLM 失败 | 观察 = 前 16k 截断文本 | 无 |
| 提炼 JSON 解析失败 | 观察 = 前 16k 截断文本 | 无 |

原则：所有新 LLM 调用均 guarded，循环永不因新功能崩溃；每条失败路径都有日志。

## 9. 测试策略

- 纯函数（unittest，无异步）：`_items_digest`（USPTO/非 USPTO/超 20 条/空）、`format_distilled`、args q 键覆盖（q/query/params 三形态 + JSON 字符串内嵌）
- 异步（`IsolatedAsyncioTestCase` + AsyncMock）：改写缓存只调一次、改写失败保持 args、下载+提炼成功路径、下载失败观察、提炼失败降级 16k
- 共享抽取回归：既有长任务测试全量（基线 369 passed / 26 failed / 9 errors 均为预存环境问题，零新增）
- 测试运行：`PYTHONUTF8=1 python -m pytest ... -q --continue-on-collection-errors`（系统 python C:/Python314）

## 10. 前端影响

**零新开发**：步骤条/折叠时间线已实现（待部署）；`fetch_patent_spec` 的 step 行自动显示「正在下载「专利号」的说明书」；回答为普通流式文本；完整搜索列表走既有 raw_items 展示。前端部署（Cloudflare Pages）完成后 Q1 的步骤可视化问题自然解决。

## 11. 范围外（明确不做）

- 聊天路径不做相关性闸门/翻页/去重（长任务管线 Phase 1 已覆盖长任务路径；聊天路径靠改写 + 摘要观察 + LLM 判断，保持轻量）
- 不做 CNIPA 下载（项目只做美国专利）
- 不做全库语义检索（Phase 2 按需）
- 不改长任务管线的路由与行为（除下载实现抽取为共享模块）

## 12. 验收标准

**多领域基准查询集**（每个领域按统一指标打分；单条失败单独定位断点，不阻塞整体判定）：

| # | 查询（占位，可替换/扩充至 5-8 条） | 领域 |
|---|---|---|
| 1 | 工业在线干燥空气源提供与设备内部湿度精准控制（回归用例） | 工业除湿 |
| 2 | 电动汽车动力电池热失控预警与散热结构 | 新能源/电池 |
| 3 | 半导体工艺腔室的温度控制与晶圆加热 | 半导体设备 |
| 4 | AR 眼镜光学波导与全息显示 | 消费电子/光学 |
| 5 | 医学影像的 AI 辅助诊断算法 | 医疗 AI |

**每查询验收项**（全部检查）：

| 检查项 | 通过标准 |
|---|---|
| 清单切题率 | 回答列出的 5-10 篇专利中 ≥80% 与查询主题直接相关；无跨领域噪声 |
| 分析质量 | 每篇有一两句基于观察摘要的真实相关性分析（非套话、非编造） |
| 深挖真实性 | 追问任一篇时 agent 调用 fetch_patent_spec，回答基于真实提炼内容（发明点/权利要求要点与专利实际内容一致） |
| 改写日志 | 搜索请求的 q 为 OR 组/引号短语英文形态，无直译 filler 词 |

**整体判定**：5/5 查询达到清单切题率与深挖真实性标准；回归用例（#1）另对照此前失败基线（880 万噪声命中 → 切题清单）。

**通用回归**：全量测试零新增失败；长任务管线下载行为不变；前端部署后步骤行 + 折叠时间线正常显示。
