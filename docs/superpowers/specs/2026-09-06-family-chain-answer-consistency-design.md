# 方案：家族深挖链可解析性 + 答复一致性 + 检索结果分组

> 批次来源：2026-09-06 用户日志分析（第 18 位样本，用户 7804999401726161950，xushan1111@gmail.com，
> 2026-09-04 02:09-02:40 轨迹）产出的 5 条改进建议 + xlsx 需求列表新增需求 26。
> 对应需求列表口径：承接需求 6（号码标准化，证据 →5/18）、需求 22（同族/审查差异，证据 →3/18）、
> 新增需求 26（答复文案与当轮真实检索结果一致性）。
> 本文为设计 spec，执行拆分为单独 plan（沿用 2026-09-05 需求#7 流程）。
> v2（2026-09-06）：合并双路对抗评审（设计评审 + 代码事实审计）结论，修订记录见 §12。

---

## 1. 背景与问题

### 1.1 问题一：PCT/国际申请号全链路不可解析（建议①，用户 7804 现场）

2026-09-04 02:39:24 用户提问「分析 PCTUS2021059064 及其全球同族申请的审查差异」。日志显示：

- `number_parse — [{"display": "US2021059064", "country": "US", "id_type": "ambiguous", "confidence": "medium"}]`
  把 `PCTUS2021059064` 剥成美国纯数字两口径；
- claims/单专利 detail 路由 `GET /applications/2021059064/documents` 403 + `POST /applications/search` 404×2 →
  `claims fetch failed … Could not resolve USPTO application for id 'PCTUS2021059064'`；
- ReAct 家族关键词检索式空转 ~16s（02:39:44 → 02:40:00）才触发 long_task families；
- Celery 内 families Phase 0 EPO family lookup 全格式 404（`CLIENT.InvalidCountryCode`），
  02:40:09 两次 `long_task:fail`。

勘察 + 代码事实审计实证（位置均以 feat/baiten-dual-source 为基线）：

| # | 事实 | 位置 |
|---|---|---|
| A1 | 解析器无 PCT/WO/国际申请号分支；`_PREFIXED_RE` 的 `WO` 命中只进 `_external()`（unsupported + 空 lookups）；`PCTUS…` 因 `T` 为 word char 破坏 `\bUS` 边界不触发前缀，只剩裸数字（**实测复现**：`PCTUS2021059064` 只扫出 bare `2021059064`；`PCT US2021 059064`/`PCT/US2021/059064` 带空格形态会落 US prefix —— 设计须先统一剥离 PCT 头再解析） | `sources/patent_number_parser.py` 204-214/226-229/313/319-357 |
| A2 | `_classify_bare` 对 ≥9 位任意数字无长度上限守卫，无条件产出 `US ambiguous` 兜底（实测 10/11 位均落此）；`format_number_guidance` 只收 CN/US 候选 | 同文件 336-357/427-430 |
| A3 | 分类 prompt 只教 "USPTO 7-8 位数字 / CNIPA 20XX…" 两种格式，PCT 形态按 US 数字处理 | `api_routes/core.py` 296-310 |
| A4 | `resolve_application_number` 剥前缀取纯数字后直查 documents → search → 抛错携原始带前缀 id；claims/spec 为前端 detail 路由（`/patent/{source}/{id}/claims|spec`），与 ReAct 环无关，**不受任何预检门约束**；documents 非 200 一律转 search（403 与合法 0 命中无分叉，仓库无 403 特判） | `sources/uspto_download.py` 174-201/88-89；`api_routes/patent_detail.py` 596-601/648-651/722-758 |
| A5 | EPO OPS 语义 = 按 DOCDB **公开号**查同族（`…/family/publication/docdb/{pub_number}/biblio`）；families Phase 0 候选把原申请号/申报号递归送检 → 全军 InvalidCountryCode；**Phase 0 失败即整体硬失败，无降级腿** | `sources/long_task/patent_family.py` 34-36/97-134；`celery_worker.py` 1937-1988 |
| A6 | CN/EP/JP examination resolver 对非本地号统一委托 `epo_client.lookup_family`，均无 PCT 前置处理 | `china_examination.py` 141-223 / `epo_examination.py` 89-172 / `japan_examination.py` 56-135 |
| A7 | 确定性预路由 `_match_long_task_intent` 只对带 `knowledge.question` 的类型 3 条目生效；内置通用 `patent_deep_analysis`（knowledge=None）**刻意被护栏排除**（注释「宁可命中不可漏判 → 会把 EVERYTHING 路由进深任务」），留在 ReAct 环内 —— ~16s 空转的机制根因 | `general_agent.py` 2061-2099；`react_tools.py` 346/588-606/846-855/2997-3001 |
| A8 | 长任务提交前无任何可解析性预检（chat 主链 core.py 1337-1471）；`submit_long_task` 的 `_normalize_submit_patent_id` 仅弱校验（family 分支 ≥6 位含数字即过，`PCTUS2021059064` 能通过）；`retry_long_task` 复跑前同样无门 | `api_routes/long_task.py` 128-160/300-308/338-417 |
| A9 | resume 隐患：`_dispatch_from_mysql`（暂停任务恢复）**硬编码 `execute_patent_analysis.delay`**，不读 task_type，families 若经 resume 会错 executor | `api_routes/long_task.py` 19-55（delay 在 49-50；调用点 470-489） |
| A10 | 仓库**全链无任何 WO 号成功送 EPO 的先例/测试**（docstring 示例全为 US）——WO docdb 直猜是否被 OPS 接受是真实开放项 | `patent_family.py` docstring 97-98 |

### 1.2 问题二：答复文案与当轮真实检索结果脱节（建议② = 需求 26，用户 7804 两轮）

同一现场 02:09 与 02:32 两轮，`patent_search_result`/`patent_search_notes` 显示真实命中
`us_hits=20 cn_hits=10 total=30`、`Baiten 10 hits`（第二轮自动补跑后 cn 20），用户面板照此落库并
据面板下载 CN118837496A；但 LLM 终稿两轮均断言「中国专利检索接口账号已过期 / 未能返回有效候选」。

勘察 + 审计实证：

| # | 事实 | 位置 |
|---|---|---|
| B1 | 内置双源工具（`_run_patent_search`）返回给 LLM 的 observation = `_items_digest` **无分源计数、无逐通道 notes** 的候选行列表；notes（USPTO N hits / Baiten 0 gateway / failed…）**只进日志**，注释明文「不拼进用户可见 observation (2026-09-01)」。**附注**：截断（>20）时 `_items_digest` 已有 `共 N 条` 尾行（661-664，非分源），单 CN 号命中 US citing 时带附注（2740）——新计数行须用独立前缀（设计用 `[本次检索]`）避免语义叠加 | `react_tools.py` 614-671/2652-2661/2710-2743/660-664 |
| B2 | 动态（USPTO 池）路径的 observation 头部带「检索结果（N 条…note…）」计数 —— 内置与动态路径存在**能力鸿沟** | `react_tools.py` 3124-3159 vs 2743 |
| B3 | ReAct 循环两次 `llm_call` 之间无任何统计/notes 注入 `messages` | `sources/agents/react_loop.py` 81-196 |
| B4 | 「账号已过期」叙事在当轮工具观察中**不存在**（代码库无该硬编码串、佰腾内置失败仅折叠进 notes）→ 唯一来源 = 历史轮失败叙述经 `_build_previous_conversation_block` 原样进入 system prompt | `general_agent.py` 2003-2043 |
| B5 | 前端面板由独立 faithful-transcriber LLM 按 pruned 批次 JSON 渲染（与终稿 LLM 分离），故面板有 CN 而终稿否认 —— 用户视角「面板与正文打架」 | `general_agent.py` 2328-2415/2452-2455 |

### 1.3 问题三：家族/审查差异失败无挽回引导 + 事件双计（建议③）

- 失败会话消息模板机械重复 error（「EPO family lookup failed for all formats (…)」），
  无「该给什么正确格式号 / 可提供 WO 公开号或国家阶段申请号」类引导；
- 同一失败**触发两次 `long_task:fail`**：executor 内 `set_task_failed`（1981，analytics 无幂等）
  + 外层 `_notify_terminal_failure → set_task_failed`（2954-2958 / status_manager.py:196）；
- executor 内联 `_update_mysql_progress('failed',0)`（1986）**是当前唯一把 MySQL 置终态的调用**；
  外层 returned-failed 分支只 notify 不落 MySQL（对比：其余 executor 异常分支均为
  `_notify_terminal_failure + _update_mysql_progress('failed',0)` 成对）。

### 1.4 问题四：技术簇分组缺失（建议⑤，P2）

宽跨域概念组检索（污染×溯源×异常定位）的美库结果混入 IT 故障诊断 / 碳核算等噪声簇；
语义重排为单列表单调重排序，无聚类/分组结构（`semantic_rerank.py fuse_ranking`）。
候选字段边界：Baiten 项 `cpc_codes=[]` 且无 abstract（`react_tools.py` 1949-1963/2462-2480），
USPTO 项经 `build_candidates` 才有 abstract。现有唯一聚类骨架 =
`grounded_interpretation.py` 按查询解读维度对 top 候选做 Flash 聚类（≥15 scored，仅动态链触发）。

### 1.5 现状中已具备、本方案复用的资产

- 需求#7 基建：`append_task_message`（event=created/completed/failed 会话消息 + hydrate 补历史）、
  `set_task_failed`、锚点 `sess:{session_id}:anchor`（24h 滑动）——失败引导消息可直接走既有会话消息链展示；
- families 成功路径的 `_resolve_us_app_to_pub_number`（USPTO search 反查 grant/pub/app 三元组，
  worker 1815-1920）——translator 对 **US 号**的 docdb 候选可收敛复用；
- 检索管道计数/notes 已逐通道生成（仅日志层）——抬升进 observation 成本低；
- `react_tools.py` 已有家族意图词表 `_FAMILY_INTENT_KEYWORDS_ZH`（355-357）与 core 分类器
  families 场景关键词 —— 预路由直接复用，不另造词表；
- chat_fallback 既有「降级走常规 chat 且剥离 long_task」通道（core.py 1290-1333）——预检失败
  备选呈现路径。

> **评审修正（v2）**：原 §1.5 将 families CN 腿的 `GooglePatentsClient` 列为「PCT→WO 反查可复用
> 通道」——审计证实**该前提不成立**：`GooglePatentsClient`（sources/google_patents_client.py:52-390）
> 只有按已给定公开号抓单一详情页的方法（query_claims/query_description/query_basic_info/…），
> 无任何按申请号反查同族/成员的方法；families CN 腿的 CN app/pub 全部来自 EPO `family`
> 返回的 `get_representative('CN')`，从未做过 PCT→WO 反查。**反查若做须新增方法**（成本与风险上
> 升），本方案将其降为实验通道（默认关，见 §5.2/§10）。

### 1.6 缺口清单（本方案要补的洞）

1. 解析层不认识 PCT/WO/国际申请号，裸数字无长度守卫（A1/A2/A3）；
2. 无「号 → 各数据源可执行口径」翻译层与可解析性判定（A5/A6/A8）；
3. 前端 claims/spec detail 路由把任意外文号剥前缀送 USPTO 空转 403/404，且不受任何门约束（A4）；
4. 家族意图无预路由 → 空转（A7），且预路由扩展须绕过既有 knowledge=None 防呆护栏（A7 注释）；
5. 失败消息无引导、事件双计、MySQL 终态依赖 executor 内联（问题三）；
6. 内置检索 observation 无分源计数（B1/B3）→ 终稿无法与真值对齐；
7. 语义重排结果无技术簇分组（问题四）。

---

## 2. 目标、非目标与验收

### 2.1 目标

- **G1（建议①）**：输入 PCT/WO 国际申请号时，系统能识别其形态并给出规范读法；
  **分层可解析路径**（按实证强度，见 §5.2）：对确定性可解析案件（US grant/pub、WO 公开号由用户
  提供/反查确认）成功进入家族任务；对不可解析案件在**提交前**秒级判定并给出可操作引导。
  PCT→WO 自动反查为实验通道，未实证前默认按引导路径交付（fail-closed）。
- **G2（建议①③④）**：家族/单专利任务不可解析时**不落 Celery 空转**——提交前预检拦截并以
  **模板引导答复**直接结束本轮（**不创建任务行、不派 celery、不产生失败事件**，无任务噪音）；
  **claims/spec 前端 detail 路由同步加格式门**（外文/不可解析形态不再剥前缀送 resolve 空转，
  给出号码引导；US/CN 合法形态零影响）；worker 侧执行失败的会话消息带格式引导与重试路径；
  同因失败 analytics 单发、MySQL 终态与 Redis 单点一致；KB 无家族项用户的「pct/wo 候选 + 家族分析
  意图」请求不再经 ~16s 关键词空转。
- **G3（建议② = 需求 26）**：内置双源检索的终稿上下文可见分源权威计数行（`[本次检索]` 前缀，
  与既有截断「共 N 条」语义隔离）；system 层有「以当轮 observation 计数为准」通用约束。
- **G4（建议⑤，P2）**：语义重排后结果支持按技术簇分组标注（flag 门控，先 observation 层，不做面板改版）。

### 2.2 非目标（YAGNI）

- 不做 WO/PCT 审查档案深度数据接入（WO/IA 检索报告正文），仅保证「号可识别 + 可解析案件能查 +
  不可解析案件秒级引导」；
- 不做前端面板改版（簇标注不进面板 UI；detail 路由仅加后端格式门，不改前端）；
- 不做法律状态数据源扩充（需求 18 另行立项）；
- 不做「全库号码字典」：翻译层只覆盖本系统三种目标源（USPTO/EPO OPS/佰腾 CN）需要的口径；
- 不做历史轮事实核验引擎（全量回答校验超出范围）；只做观察层真值注入 + 规则约束；
- **不改动知识库无关的通用 deep-analysis 预路由护栏**（A7）——只新增「pct/wo/unsupported 候选
  且家族分析意图」的窄门（§5.5b），不放开 knowledge=None 通用项；
- resume 错 executor（A9）：列入 Phase A 收尾修复（`_dispatch_from_mysql` 按 task_type 分派），
  修复面小不单独立项。

### 2.3 验收

自动化（单测/集成）：
1. parser 表驱动：PCT/WO 各形态（`PCTUS2021059064` / `PCT/US2021/059064` / `PCT US2021 059064` /
   `US2021/059064` 语境 / `WO2021059064` / `WO2021/059064A1`）产出 `id_type ∈ {pct, wo}` 与规范 display；
  现有 CN/US 回归零破坏 —— **含 9 位 CN 公开号（`117941643` → CN pub，3 个既有用例）与 12/13 位
   CN 申请号**（守卫必须位于 CN 分支之后）；≥9 位非 CN 裸数字 → unsupported（不再 US ambiguous）；
2. 预检门：mock celery，`patent_id` 不可解析（PCT 且翻译失败 / 未知前缀）→ 断言**未创建任务行、
   未调用 `execute_family_analysis.delay`**，本轮以含格式引导的模板答复收尾，且**无
   `long_task:fail` 事件**；可解析（US 授权号）→ 照常提交；translator 抛异常 → 放行提交（旧行为）；
3. **detail 路由门**：`/patent/uspto/claims` 带 PCT/WO 号 → 断言**未调用**
   `resolve_application_number`（返回引导错误）；带合法 US 号 → 行为与现状一致；
4. 失败终态一致性：executor 返回 failed dict → 外层单点写 Redis failed + MySQL failed 各一次，
   analytics `long_task:fail` **仅 1 次**；
5. 内置检索 observation：候选 >0 时 digest 尾部含 `[本次检索] US n 条 · CN n 条` 行；**空候选不加行**；
   既有 `_items_digest` 单测（`test_react_tools.py` `共 30 条`/截断语义、`test_dual_patent_search.py`
   `共 50 条`）零破坏；
6. 簇分组（Phase C）：≥阈值候选 mock → observation 含 `技术簇` 标注；<阈值/异常 → 无标注、不抛错。

轨迹复跑（部署后人工核对日志，参考 09-04 现场）：
- [ ] 输入「分析 PCTUS2021059064 及其全球同族申请的审查差异」：日志出现 PCT 识别（非 US ambiguous）；
- [ ] KB 无家族项时该请求**不再经 claims 403/404 + 关键词 16s 空转**（预路由窄门或预检引导先达）；
- [ ] 反查通道（flag 默认关，Phase A 后实现）未解锁前：本轮直接以「请提供 WO 公开号或美国国家阶段
  申请号」类引导答复收尾（无任务创建、无 `long_task:fail`）；用户补充 WO 号重问 → 走通解析路径；
- [ ] 反查通道 spike 已证可行（2026-09-06：PCTUS2021059064 → WO2023075806A1 唯一命中）；WO→EPO
  docdb kind-less 接受度 UAT（服务器凭据试发 `WO2023075806` vs `WO2023075806A1`）确认后，
  families 以合法 docdb 号进入 EPO 并产出家族结果；
- [ ] 用户改提供 WO 公开号（如 WO2021/xxxxx）重问 → 解析路径走通 EPO（OPS 接受 kind-less 与否
  一并实测）；
- [ ] detail 路由对 PCT 号点击 → 返回号码引导而非 403/404 空转；
- [ ] 概念组检索轮终稿如提及检索概况，与 `patent_search_notes` 当轮真值一致（不再声称 CN 不可用）；
- [ ] （Phase C）observation/日志出现技术簇标注。

---

## 3. 架构总览

新增确定性「号解析 → 翻译 → 可解析性判定」三层，插在现有各入口之前；其余均为对既有路径的最小注入。

```text
用户输入(含 PCT/WO)
   │
   ▼
[1] patent_number_parser 扩展        ← 识别 pct/wo/裸数字长度守卫；PCT 头先剥离再解析；不破坏 9/12/13 位 CN
   │
   ▼
[2] patent_id_translator(新模块)      ← 号 → {每源候选口径} + verdict{resolvable/unresolvable}
   │  ├─ 本地格式正则（CN/JP/EP/US pub-grant；WO 公开号直配）
   │  ├─ EPO DOCDB 公开号候选（US grant/pub 复用 _resolve_us_app_to_pub_number；WO pub）
   │  └─ PCT→WO 自动反查 = 实验通道（flag 默认关，需新增 GooglePatentsClient 检索/同族解析方法）
   ▼
[3] 预检门(提交前 × 四入口)           ← chat 主链(INSERT 前)、submit_long_task、retry_long_task、
   │                                    前端 detail 路由(claims/spec 轻量格式门)
   │  ├─ resolvable        → 照常 celery（翻译产物注入 families Phase 0 候选）
   │  ├─ unresolvable      → 模板引导答复直出：不建任务行、不派 celery、无失败事件
   │  │                      （chat=答复收尾 / submit=4xx 带引导 / retry=拦并提示 / detail=引导错误）
   │  └─ translator 异常    → 放行（保持旧行为，worker 兜底）
   ▼
[4] 失败单点出口 + 引导模板           ← executor 不再写 failed；外层 returned-failed 分支
   │                                    notify + _update_mysql_progress('failed',0) 成对；
   │                                    failure_guidance() 按结构化原因给可操作建议
   │
[5] (P2) 语义重排后技术簇分组          ← flag 门控，observation 层标注
```

答复一致性（需求 26）为横向注入，不依赖 1-4：

```text
_run_patent_search 返回 digest 时（候选 >0）
   └─ digest 尾部 += "\n[本次检索] US n 条 · CN n 条（合并共 m 条，按相关度排序）" + notes 收敛行(≤300 字)
      —— 独立前缀，与既有截断「共 N 条」隔离；空候选不加行
loop_system_guidance += 通用约束句（见 §5.5a）
```

---

## 4. 端到端数据流

场景：用户问「分析 PCTUS2021059064 及其全球同族申请的审查差异」（chat 主链）

```text
query_stream ─ create_agent
   ├─ parser：剥离 PCT 头 → country/office=US(受理局)、year=2021、serial=059064
   │          → id_type=pct, display="PCT/US2021/059064"
   ├─ [预路由窄门] pct 候选 && 家族分析意图词命中 && KB 无家族 type-3 项
   │      → 直接返回 intent families（不再 claims 403/404 + 关键词 16s 空转）
   │      （KB 有家族 type-3 项：维持既有预路由；检索型"同族"提问：不命中窄门，走检索）
   ▼
core _classify/_prepare → scenario=families, patent_ids=[PCTUS2021059064], source=auto
   ▼
[预检门] translator.translate("PCTUS2021059064", scenario=families)
   ├─ 反查 flag ON 且确认到 WO 公开号（spike 已证可行）→ 候选集 {epo_docdb:"WO2023075806A1…"}
   │     → 提交 celery，families Phase 0 改用翻译候选（无直猜通道）
   ├─ 均未确认 → verdict=unresolvable → 本轮以模板引导答复收尾（INSERT 前拦截，不建任务）：
   │     「我识别到 PCT/US2021/059064 是国际申请号，家族分析需要其公开号形态。
   │      请提供 WO 公开号（如 WO2021/xxxxx）或美国国家阶段申请号，我将直接为您分析；
   │      也可改为按申请人/优先权检索。」
   │     （无任务行、无 long_task:fail；答复入会话，下轮用户带新号重问即走通）
   └─ translator 异常 → 放行提交（行为与现状一致，worker 兜底）
```

detail 路由场景：用户对结果行点「权利要求」（PCT 号）→ 格式门直接回引导错误，不经 resolve。

检索轮（概念组检索）：内置 dual 工具返回 digest + `[本次检索]` 分源行 → ReAct observation 含真值；
后续追问「中国专利是不是查不了」类 → 模型可见当轮 CN 计数，不再沿用历史叙事。

---

## 5. 行为细节

### 5.1 号码识别扩展（改 patent_number_parser.py）

新增两种 `id_type` 与对应分支（保持现有构造器与字段形状不变，向后兼容）。

**解析顺序（关键，防连写/空格形态分裂）**：token 扫描时先匹配 `PCT` 头再进入既有前缀/裸号逻辑；
`PCT` 头形态统一为 `PCT[/\s]?{RO}[/\s]?{YYYY}[/\s]?{NNNNNN}`（RO 两字母受理局），剥离后按
`{RO}{YYYY}{NNNNNN}` 校验（年份 4 位 + 序号 ≥5 位），避免 `PCT US2021 059064` 落 US prefix
（A1 实测）。`WO` 前缀分支从 `_external()` 提升为真解析。

| 输入形态（示例） | id_type | display（规范读法） | lookups | 引导 |
|---|---|---|---|---|
| `PCTUS2021059064` / `PCT/US2021/059064` / `PCT US2021 059064` | `pct` | `PCT/US2021/059064` | 空 + `meta{office:"US"}` | §5.2 判定 |
| `WO2021059064` / `WO2021/059064` / `WO2021/059064A1` | `wo` | `WO2021/059064` | `["WO2021059064"]`（docdb 候选，kind 可省待实证） | §5.2 判定 |
| `PCT/CN2021/xxxxxx` / `PCTCN…`（受理局 CN） | `pct` | `PCT/CN2021/xxxxxx` | 空 + `meta{office:"CN"}` | 国家阶段引导走 CN |
| `{RO}2021/059064` 类（US 段 10 位裸数字，见下） | 见守卫 | — | — | — |

规则：
- 显式 PCT 前缀优先于裸数字兜底（含空格/斜杠形态先剥离统一）；
- `_classify_bare` 增加**长度守卫**，且**必须置于 CN 分支之后**：先保留 12/13 位 `19|20` 开头 CN
  申请号分支与 **9 位 `1` 开头 CN 公开号分支**（`117941643` 等既有用例依赖），其余
  **≥9 位裸数字 → `id_type=unsupported`**（reason：「数字长度不符美国号段，疑似残缺国际申请号/
  含 PCT/WO 前缀，请补全后再查」），lookups 为空——不再产 US ambiguous 白跑 USPTO；
- `format_number_guidance` 扩展接受 `pct/wo/unsupported` 候选，输出对应引导句；
- `decide_number_source`：pct/wo 不强制单源（维持 auto/dual）。

> 注：PCT 号 `country` 填 `WO`（文献国别语义，EPO/前端可识别），受理局入 `meta.office`。
> 具体消费点核对（parser 输出变动传导）：`decide_number_source`、`format_number_guidance`、
> 检索路由国别判定、detail 路由格式门（§5.3）——实施时逐一 grep 消费点，避免只改解析不改消费。

### 5.2 翻译层与可解析性判定（新模块 sources/patent_id_translator.py）

```python
@dataclass
class TranslateResult:
    patent_id: str
    verdict: str            # "resolvable" | "unresolvable"
    candidates: dict        # {"epo_docdb": [...], "uspto": [...], "cnipa": [...]}  可执行口径串列表
    reason: str             # 中文判定说明
    confidence: str         # high/medium/low

async def translate(patent_id: str, scenario: str = "",
                    reverse_lookup: bool = False) -> TranslateResult
def verdict_of(patent_id: str, scenario: str) -> str | None   # 同步快速门（detail 路由等非 async 段）
```

判定表（按 scenario 目标源；`reverse_lookup` 即 PCT→WO 实验通道，默认 False）：

| 输入归类 | uspto 可查 | epo docdb 可查 | cn 可查 | verdict/动作 |
|---|---|---|---|---|
| US grant/pub（8 位内） | ✓ | ✓（`_resolve_us_app_to_pub_number` 产物复用） | — | resolvable |
| US 申请号（series/serial） | ✓（documents 反查） | 仅当有 grant 产出 | — | resolvable（epo 缺则标 low confidence，仍放行） |
| CN 申请/公开号 | —（引导改源） | 经 CN 成员 | ✓ | resolvable（regional executor 自管） |
| **WO 公开号** | ✗ | ✓（docdb `WO{YYYY}{NNNNNN}`，kind 可省待 UAT 实证） | — | **resolvable*（带实证标记）** |
| **PCT（受理局 US/任意）** | ✗（无国家阶段号） | ✗（**直猜已实证证伪**，2026-09-06，见 §11——不再构造直猜候选） | 经国家阶段 | reverse_lookup=False → **unresolvable + 引导**；=True 且反查确认 WO pub → resolvable（evidence=reverse_lookup） |
| 未知前缀/≥9 位裸数字 | ✗ | ✗ | ✗ | **unresolvable + 引导** |
| translator 内部异常 | — | — | — | 抛给上层 → 放行旧行为 |

实现要点：
- EPO docdb 候选构造（无网络）：US grant `f"US{n}"`；用户**直接提供**的 WO 公开号（id_type=wo）
  `f"WO{YYYY}{NNNNNN}"`（evidence=local）。**不再对 PCT 构造直猜候选**——2026-09-06 服务器实证：
  真实公开号 WO2023075806A1（2023-05-04 公布）≠ 直猜 WO2021059064（ABB 充电案，2021-04-01），
  恒等命题证伪；PCT 自动解析只走反查通道（evidence=reverse_lookup）；
- **PCT→WO 自动反查（flag `REACT_PCT_REVERSE_LOOKUP` 默认 `"0"`，实现排 Phase A 后）**：
  2026-09-06 服务器 spike **已实证通道可行**——XHR 端点 `https://patents.google.com/xhr/query?url=q%3D{PCT号}`
  对 `PCTUS2021059064` 返回唯一命中 `WO2023075806A1`（filing_date 2021-11-12 与申请号吻合；
  JSON 直接含 publication_number + family_metadata，比页面抓取稳定）；实现 = 新增
  `GooglePatentsClient` 检索方法（该类现无任何反查方法，属**新增能力**），0 条/多条无法消歧 →
  unresolvable（fail-closed）；
- **双反查差异注**：`resolve_application_number`（uspto_download.py，PEDS documents 直查）与
  `_resolve_us_app_to_pub_number`（celery_worker.py，USPTO search 反查 grant 三元组）是**两套不同
  语义的反查**；translator 只收敛后者进 docdb 候选，前者保持 detail 路由既有用法——勿合并抽错对象；
- `_resolve_us_app_to_pub_number`（现 celery_worker 1815-1920）收敛为 translator 内部可调函数，
  families Phase 0 候选构造改调 translator（消除 1937-1963 内联候选扫描的重复）。

### 5.3 预检门（改 core.py / api_routes/long_task.py / api_routes/patent_detail.py）

触发点（四入口同门，共享 `verdict_of`/`translate`）。**裁决（2026-09-06）：预检拒一律不创建任务**——
chat 以模板引导答复收尾、submit/retry 以错误响应带引导返回、detail 以引导错误返回；均不产生
任务行、celery 派发或 `long_task` 失败事件（无统计噪音，无"从未执行的任务"记录）：

1. **chat 主链**：`core.py` scenario 判定（1290 区）之后、会话/任务 INSERT（1376）**之前**拦截；
   仅对 `scenario == "families"` 执行（其余单专利场景见 A6 共享门）；
   verdict=unresolvable → **模板引导答复直出**：本轮不再走长任务创建，以 assistant 引导文本收尾
   （复用 chat_fallback 的「降级直答」通道形态，但内容为 `failure_guidance` 模板而非重跑检索），
   SSE 正常 end；答复随既有消息存储入会话，用户下轮补 WO 号/国家阶段号重问即正常走通；
2. **submit_long_task**：`long_task.py` `_normalize_submit_patent_id`（128-160）升级为 verdict 判定
   （弱正则保留作为 fallback），`execute_*_analysis.delay`（300-308）前同门 —— unresolvable 返回
   4xx + 引导（不建任务行）；
3. **retry_long_task**（338-417）：重试前同门 —— unresolvable 拒绝重试并返回引导错误（任务保持
   原 worker 失败态；"重试"仅对 worker 执行失败（瞬时/远端）有意义，对号不可解析无意义，故不再
   放行同号重放）；
4. **前端 detail 路由（claims/spec）**：`patent_detail.py` `_fetch_claims`/`_fetch_spec_pdf`
   （596-601/648-651）调 `resolve_application_number` 之前加**轻量格式门**：`verdict_of(id)` 返回
   unresolvable（pct/wo/unsupported/外文形态）→ 直接返回引导错误（含规范格式建议），**不剥前缀送
   resolve**；US/CN 合法形态零影响（verdict 放行）。这封堵 A4「任何外文号复现 403/404」的洞。

放行原则（§7）：verdict 判定仅拦「确定性不可解析」；`needs_external` 语义并入 unresolvable（引导）；
translator 异常 → 放行保底。

**A6 共享门（裁决：必做，计入 Phase A）**：三个 examination resolver（CN/EP/JP）委托
`lookup_family` 前各加一行 `verdict_of()` 快速门（判定/引导全复用 translator 与 failure_guidance，
每处 ~1 行）——消灭"chat 有引导、submit/表单入口白跑"的入口间不一致；仅**各源本地号格式细则表**
（JP 公开号 4 段式、EP 申请号与公开号同长度消歧等）超预算时 DEFER（放行原则兜底），验收口径相应
注明「共享门全覆盖、细则表按预算」。

### 5.4 失败单点出口、幂等与引导模板（改 status_manager.py / celery_worker.py）

- **单点出口（MySQL/Redis 一致性，评审修订）**：
  - executor 内 families Phase 0 fail 分支（celery_worker.py:1980-1988）**不再自行
    `set_task_failed`/`_update_mysql_progress`**，统一 `return {'status':'failed', ...}`；
  - **外层 returned-failed 分支（2954-2958）升级为终态单点**：`_notify_terminal_failure(...)` **与
    `_update_mysql_progress('failed',0)` 成对执行**——与其余 executor 异常分支惯例
    （notify + mysql 成对）一致；`notify_terminal_failure` 内部保持单次 `set_task_failed`；
  - 由此 `long_task:fail` 自然单发（executor 不再写 failed），**不再需要**「error_message 相等去重」
    逻辑；删除 §7.5 旧「并发窗口接受」条款，改为「单 setter 自然单发」；
- **预检拒不产生任务/事件（裁决，2026-09-06）**：预检判定不可解析时**不创建任务行、不发
  `long_task:fail`**（origin 概念不再需要）；`failure_guidance` 模板同时供三条路径使用——
  chat 主链/submit/retry 的**引导答复与错误响应**（呈现为普通答复或 4xx 文案），以及 worker 执行
  失败时 `notify_terminal_failure` 的会话消息（呈现为任务失败 + 面板重试，此路径保持既有语义）；
- **引导模板**：三条路径共用新纯函数 `failure_guidance(task_type, reason_code, error) -> str`
  （status_manager.py），**reason 分类用结构化 reason_code（translator verdict 直接产码，worker 错误
  由调用点归类后传码），不做子串匹配**：

```text
reason_code                                  → 追加引导段
ERR_UNRESOLVABLE_ID(translater verdict)      → 「EPO 同族查询需要公开号格式。请提供：
                                               · WO 公开号（如 WO2021/xxxxx）
                                               · 或美国授权/公开号（如 US12506212）
                                               · 或国家阶段申请号；可回复原问题并附上新号」
ERR_EPO_REMOTE(5xx/timeout/credentials)      → 「外部服务暂时不可用，请稍后在任务面板点击重试」
ERR_OTHER                                   → 维持现模板 + 「可在任务面板点击重试，或重新描述需求」
```

引导段为**通用句**（无用户提问词）。

### 5.5 答复一致性（建议② = 需求 26，改 react_tools.py / general_agent.py）

**a. observation 计数行 + 规则句**
- `_run_patent_search` 返回 text 前（react_tools.py:2718 附近），**候选 >0 时** digest 尾部追加
  权威行（≤ 200 字，**独立前缀不与截断「共 N 条」语义叠加**）：
  `"\n[本次检索] US {us_hits} 条 · CN {cn_hits} 条（合并共 {total} 条，按相关度排序）"`，
  并将原「只进日志」的 notes 收敛成一行（≤ 300 字）附其后；**空候选不加行**（保持
  `_items_digest([])` 契约与既有「两个数据源均未返回结果」默认文本）——改动仅限内置 dual 路径；
- `_loop_system_guidance()`（general_agent.py）追加通用规则句：
  「检索工具的 observation 末行『本次检索』是对当次调用结果的权威计数；若与历史对话中的
  检索/数据源状态描述不一致，以当次 observation 为准。」——纯规则句，不固化任何提问词；
- 既有 digest 单测（`共 30 条` 截断语义、`_items_digest([]) == ""`）零破坏（计数行挂在
  `_run_patent_search` 层，不改 `_items_digest` 本体）。

**b. 家族意图预路由窄门（建议④，评审修订 —— 不放开 knowledge=None 护栏）**
- 现状护栏（A7）：`_match_long_task_intent` 只对 type-3 KB 家族项生效，内置通用 deep 项
  knowledge=None 被刻意排除——**维持不变**；
- 新增**窄门**（general_agent.py 预路由区块 2061-2079 内、KB 项过滤之后）：仅当
  `parser 产出 pct/wo/unsupported 候选`（§5.1）**且** 命中家族**分析**意图（复用
  `_FAMILY_INTENT_KEYWORDS_ZH`（react_tools.py:355-357）+ 审查分析动词（审查/授权/驳回/差异/同族/家族），
  二者交集判定）**且** KB 无家族 type-3 项命中 → 直接返回 families intent（消灭 16s 空转）；
- **防误路由**：纯检索型问法（如「检索与 X 同族且美国已授权的专利」，有"检索/查找"无审查分析
  意图）**不命中**窄门——落检索路径；误判风险接受度与护栏理由一致，窄门条件在实现时表驱动测试
  （见 §9）；
- KB 有家族 type-3 项的用户维持既有预路由路径（本就无 16s 问题）。

### 5.6 技术簇分组标注（改 react_tools.py，P2，flag 门控）

- 新 flag：`REACT_RESULT_CLUSTER_ENABLED`（默认 `"0"`）。开关开时在 `_rank_pending_pool`
  rerank 之后（1225-1236 后、截断 TopN 前）执行 `cluster_ranked(ranked, lang)`；
- `cluster_ranked`：候选 ≥ `REACT_RESULT_CLUSTER_MIN`（默认 12，命名对齐 REACT_* 阈值惯例）时，
  用 Flash 一次性对 TopN（复用 `MAX_PATENT_LIST_ITEMS` 截断列表的 id+title+applicant，
  单次调用、超时容错）产出 2-4 个簇标签 + 簇内 id 列表；任何异常返回 `None`（保持无簇行为）；
  **与 grounded 链门槛割席**：grounded 触发在动态 USPTO 链且 ≥15 语义分，本簇门在 `_rank_pending_pool`
  通用链且按候选数 ≥12，两者独立，不复用对方阈值；
- 输出：observation digest 头部插入 `技术簇：{① 标签 (n 件): id 前缀…; ② …}` 行（≤ 300 字）；
  候选结构不改（前端面板改版为非目标）；
- 输入边界：Baiten 项无 abstract/cpc → 聚类只用 id/title/applicant 三字段，不做跨源特征融合；
  中文标题聚类由 Flash 承担，不做本地分词。

---

## 6. 数据结构精确定义

### 6.1 parser 输出扩展（向后兼容增量）

```text
现有 dict 形状 + 字段:
  id_type: "pct" | "wo" | "unsupported"(新增取值，原取值不动)
  display: "PCT/US2021/059064" 等规范读法
  meta:   {office: "US"}   (仅 pct，记录受理局；无 meta 时省略)
  lookups: pct/wo 分支按 §5.1 表；unsupported 分支为空列表
```

### 6.2 translator 结果（新，内存态，不落库）

```text
TranslateResult { patent_id, verdict(resolvable|unresolvable),
                  candidates{epo_docdb[], uspto[], cnipa[]},
                  evidence: "reverse_lookup" | "local" | "", reason, confidence }   # 无 direct_guess：恒等已证伪
```

### 6.3 失败与引导（复用既有消息通道，无新存储）

```text
引导答复(预检拒, 无任务): failure_guidance(task_type, reason_code, error) 模板文本直出(chat=assistant 答复 /
                          submit/retry=错误响应 / detail=引导错误)——不建任务行、不发 long_task:fail
worker 执行失败(任务已建): append_task_message(event='failed', content=模板 + failure_guidance(...))
                          ——既有通道不变；notify 单发(§5.4)
```

### 6.4 observation 计数行与簇行（新，仅文本）

```text
内置检索 digest 尾部(候选 >0):
  [本次检索] US 20 条 · CN 10 条（合并共 30 条，按相关度排序）｜USPTO 20 hits; Baiten 10 hits
簇行(Phase C, digest 头部):
  技术簇：① 污染源解析与质谱溯源 (3 件: 18449672…)；② 系统故障根因定位 (5 件: …)…
```

---

## 7. 错误处理（全部 fail-closed 或静默降级，不破坏主链路）

1. 解析/翻译层任何异常 → 上层**放行**（走原路径，行为与现状一致），只记日志；绝不因新层故障卡死提问；
2. verdict 判定只拦「确定性不可解析」；translator 异常/低置信 → 放行（worker 兜底）；
   被拦请求仅以引导答复/错误响应呈现，**不产生任务行与失败事件**（预检拒不污染任务与指标）；
3. 计数行/簇行注入异常 → 维持原 digest 文本（try/except 包裹，单行成本）；
4. 失败引导模板构建异常 → 回退现模板（reason 截断 500 字）；
5. **失败单发**：单点出口设计（§5.4）——executor 不再写 failed，`long_task:fail` 由外层 notify
   单次发出；无双发并发窗口需接受；
6. 失败不触碰会话锚点（维持「锚 = 最近一次*完成*的长任务」语义，成功才覆盖）——文档化理由：
   本轮失败不代表旧锚结果失效；后续追问仍可引用上一圆满任务（需求#7 语义）；
7. detail 格式门与预检门只挡「确定性不可解析」形态：US/CN 合法形态必须零影响（放行）；
   误拦风险由放行原则兜底，落地后跟踪误拦率。

---

## 8. 代码改动清单（实施锚点）

| 文件 | 改动 | 对应 |
|---|---|---|
| `sources/patent_number_parser.py` | pct/wo 分支 + PCT 头先剥离 + `_classify_bare` 长度守卫（**置于 CN 分支后**）+ guidance 扩展 | G1/§5.1 |
| `sources/patent_id_translator.py`（新） | translate/verdict_of + 候选构造 + `_resolve_us_app_to_pub_number` 收敛 + 实验反查（flag 默认关） | G1/G2/§5.2 |
| `sources/google_patents_client.py` | **新增**按 PCT 号检索取 WO/同族方法（实验通道，spike 先行） | §5.2 |
| `api_routes/core.py` | 预检门（scenario 判定后 1290 区、INSERT 1376 **之前**）；引导答复直出通道（chat_fallback 变体形态） | G2/§5.3 |
| `api_routes/long_task.py` | submit/retry 同门（128-160、300-308、338-417）；**`_dispatch_from_mysql` 按 task_type 分派（A9 修复，49-50）** | G2/§5.3/§2.2 |
| `api_routes/patent_detail.py` | claims/spec 前置轻量格式门（596-601/648-651） | G2/§5.3 |
| `sources/long_task/status_manager.py` | `failure_guidance()`（供引导答复与失败消息共用）+ notify 模板；set_task_failed 增 reason_code 通道 | G2/§5.4 |
| `china/epo/japan_examination.py` | resolve 委托 `lookup_family` 前加 translator 共享门（A6 必做；细则表按预算 DEFER） | G2/§5.3 |
| `celery_worker.py` | families Phase 0：fail 分支单点出口（1980-1988 删写）、外层 returned-failed 成对补 MySQL（2954-2958）、候选构造改调 translator（1937-1963） | G2/§5.4/§5.2 |
| `sources/agents/react_tools.py` | 内置 digest `[本次检索]` 行（2718 区块）；家族意图窄门复用词表（355-357 周边）；簇分组(flag) | G3/G4/§5.5/§5.6 |
| `sources/agents/general_agent.py` | `_loop_system_guidance()` 一致性规则句；预路由窄门（2061-2079） | G3/§5.5 |
| 测试（新增/扩展） | 见 §9 | — |

---

## 9. 测试计划

- parser：扩展既有 `tests/test_patent_number_parser.py`（现含 9 位 CN 公开号用例 `117941643` 系）
  而非另起文件；新增 §2.3 表驱动用例；**回归红线：9/12/13 位 CN 分支、6-8 位歧义 US、
  带空格 PCT 形态与连写形态一致性**；
- `tests/test_patent_id_translator.py`（新）：§5.2 判定表逐行（mock 反查开/关、mock 异常、
  evidence 标记）；
- `tests/test_long_task_precheck.py`（新）：四入口预检门 —— chat 主链：mock，不可解析号断言
  **任务行未创建、delay 未调用、本轮以引导答复收尾、无 long_task:fail 事件**；submit/retry 返回
  引导错误且不建任务/不放行同号；detail 路由不调 resolve；translator 异常放行（旧行为）；
- `tests/test_failure_terminal_state.py`（新）：worker 执行失败（任务已建）→ 外层单点写
  Redis+MySQL 各一次、analytics 1 次；失败消息含 reason_code 对应建议；A6 共享门（三个 resolver
  前置门各一例：不可解析号不委托 lookup_family、返回引导）;
- `tests/test_observation_counts.py`（新/并入检索测试）：候选 >0 含 `[本次检索]` 行；空候选无行；
  既有 `共 30 条/共 50 条`/`_items_digest([]) == ""` 断言零破坏；
- Phase C：`tests/test_result_cluster.py`（新）：≥/＜ `REACT_RESULT_CLUSTER_MIN`、异常 None、行格式；
- 全量回归纪律同 09-05：新增失败 0；已知 pre-existing 失败集（batch_prompt 12f、memory 5f、
  session_api 7e、searx 3f、patent_analyzer 3f、knowledge_candidates 1f）不作为回归；
  本机 pytest 需 `PYTHONUTF8=1`。

---

## 10. 实施分期（每期独立可合入，同分支顺序合入）

> 评审修订：Phase A/B/C 均触及 `sources/agents/react_tools.py`（A 改预路由窄门区、B 改
> `_run_patent_search` 返回区、C 改 `_rank_pending_pool`），**不做并行分支**——同分支按序合入，
> 每 Phase 独立 commit + 全量回归；seller-scene 线仅并行不动。

- **Phase A（P1：建议①③④ = 家族深挖链）** 同分支内分 commit：
  - A1 解析层：parser pct/wo + 长度守卫 + guidance（含 9 位 CN 回归红线）；
  - A2 translator/verdict + `_resolve_us_app_to_pub_number` 收敛（纯重构先行可独立合入）；
  - A3 预检门四入口（chat 主链 INSERT 前**引导答复直出**、submit 4xx、retry 拦、detail 格式门）——
    均不建任务、无失败事件；
  - A4 worker 失败单点出口 + MySQL 终态成对 + `failure_guidance`（reason_code 结构化）；
  - A5 预路由窄门（消灭 16s 空转）+ A9 `_dispatch_from_mysql` task_type 分派修复；
  - A6 **共享门必做**（三 examination resolver 前置 verdict_of 一行门）+ 各源格式细则表按预算
    DEFER（放行原则兜底）；
  - 本相位直接消灭 09-04 现场四类症状（claims 403/404 空转、detail 路由空转、36s 试错、
    celery 双 fail/非终态）——不可解析案件统一以引导答复收尾；
- **Phase B（P1：建议② = 需求 26）** `[本次检索]` 行 + notes 收敛 + 一致性规则句（顺 A 同分支）；
- **Phase C（P2：建议⑤ = 技术簇分组）** flag 门控独立合入（同分支顺延）；
- **实验通道 spike（已完成，2026-09-06 服务器侧）**：实证结论——①Google XHR 按 PCT 号反查可行：
  `PCTUS2021059064` → 唯一命中 `WO2023075806A1`（XHR JSON 含 publication_number/family，比页面抓取稳）；
  ②直猜恒等命题**证伪**：`WO2021059064` 属 ABB 充电他案 → translator 不再直猜。剩余 UAT：
  真实 WO 号送 EPO OPS 的 docdb **kind-less 接受度**（服务器凭据试发 `WO2023075806` vs `WO2023075806A1`）；
- 收尾：全量回归 + 终审（code-review）→ 服务器部署 → §2.3 轨迹复跑验收 → 需求列表证据回填。

分支建议：执行从 `feat/baiten-dual-source`（= main 4efa3df）继续或拉 `feat/family-resolve-consistency`；
seller 线并行不动。

---

## 11. 风险与开放项

| 项 | 说明 | 处置 |
|---|---|---|
| **WO docdb 直猜恒等命题（已实证证伪）** | 2026-09-06 服务器样本：PCT/US2021/059064 真实公开号 WO2023075806A1（2023-05-04 公布）≠ 直猜 WO2021059064（ABB 充电案，2021-04-01 公布） | translator **不再构造直猜候选**；PCT 自动解析只走反查通道（spike 已证可行，flag 默认关，Phase A 后实现） |
| **WO kind-less 是否被 OPS 接受** | 仓库无任何 WO→EPO 先例/测试 | UAT 实测；不接受则要求带 kind（A1）形态再送 |
| **Google 反查（实验通道）** | `GooglePatentsClient` 现无按号反查方法——**新增能力**，抓取可被限流/改版 | spike 先行 + flag 默认关 + fail-closed；不阻塞主链路 |
| PCT country 语义（受理局 vs 文献国别） | parser `country` 填 WO、office 记受理局 | §5.1 注；UAT 核对前端/检索路由读取 |
| **预路由窄门误命中** | 检索型"同族"提问（无审查分析动词）若被推进家族任务 | 窄门条件 = pct/wo 候选 ∧ 分析意图词 ∧ KB 无项；表驱动反例测试（§9）；仍误命中则走既有 chat_fallback 语义 |
| LLM 一致性残余风险 | 计数行/规则句是强信号非硬保证 | 验收限缩为「计数行在场 + 复跑无文案否认」人工判定 |
| 预检/格式门误拦 | 影响面 = 本应成功的任务/取件被拒；被拦者本轮得到引导答复 | 放行原则兜底（仅拦确定性不可解析）；**预检拒不产生任务/事件**，误拦只表现为一次引导答复，跟踪误拦率与答复转成功比例 |
| 区域性单专利场景细则精度 | CN/EP/JP 各 resolve 入口的**共享门必做**（一行判定）；仅各源本地号格式**细则表**可能 DEFER | 共享门随 Phase A 交付；细则表 DEFER 期间放行原则兜底，验收口径注明「共享门全覆盖、细则表按预算」 |
| 技术簇成本 | 每次池检索 +1 次 Flash（~1s） | flag 默认关；P2 先观察后默认化 |
| 与需求#7 共存 | 锚/失败消息/回执基建直接复用 | 无冲突；失败不写锚的语义已文档化（§7.6） |
| resume 路径（A9） | `_dispatch_from_mysql` 硬编码批量 executor | Phase A5 修复（按 task_type 分派），成本低 |

---

## 12. 对抗评审修订记录与待裁决项（2026-09-06）

双路评审（对抗性设计评审 + 代码事实审计）合流后本版已修订：

| # | 评审发现 | 处置 |
|---|---|---|
| R1 | **CRITICAL**：claims/spec detail 路由不受预检门约束，任何外文号剥前缀复现 403/404 | 已修：§5.3 第四入口 + §8 列 patent_detail.py + 验收 2.3/3 |
| R2 | **HIGH**：executor 删 `_update_mysql_progress` 后 MySQL 停非终态（外层 returned-failed 只 notify） | 已修：§5.4 单点出口（外层 notify+MySQL 成对），与其余 executor 惯例一致 |
| R3 | **HIGH**：GooglePatentsClient「复用做 PCT→WO 反查」是不存在的方法假设 | 已修：§1.5/§5.2 降为「新增方法 + 实验通道 flag 默认关 + spike 先行」 |
| R4 | **HIGH**：WO 直猜混淆申请流水与公开号两套编号，判定表把推测当 resolvable | 已修：§5.2 判 PCT 默认 unresolvable+引导；**2026-09-06 spike 实证证伪恒等**（WO2021059064=ABB 他案 vs 真实 WO2023075806A1）→ 判定表移除直猜，evidence 枚举删 direct_guess |
| R5 | **HIGH**：预路由扩展撞 knowledge=None 防呆护栏（过度路由 EVERYTHING）；16s 消灭依赖 KB 前提未声明 | 已修：§5.5b 窄门设计（pct/wo 候选 ∧ 分析意图 ∧ KB 无项）；§2.1 G2/验收措辞条件化 |
| R6 | **MEDIUM**：幂等相等去重与「并发窗口接受」自相矛盾（单点出口后本无双发） | 已修：§5.4 删除相等去重，§7.5 改「单 setter 自然单发」 |
| R7 | **MEDIUM**：Phase A/B 同文件可并行表述自相矛盾 | 已修：§10 同分支顺序合入，不做并行分支 |
| R8 | **MEDIUM**：`_dispatch_from_mysql` 定位错（非 core.py，实为 api_routes/long_task.py:19-55/49-50） | 已修：§1.1 A9、§8 |
| R9 | **MEDIUM/LOW**：守卫误伤 9 位 CN 公开号（`117941643` 三用例）风险 | 已修：§5.1 守卫置于 CN 分支后 + §2.3 回归红线 + §9 |
| R10 | **MEDIUM/LOW**：失败引导子串匹配脆弱；预检拒与崩溃失败语义混淆 | 已修：§5.4 reason_code 结构化；v3 裁决后预检拒不建任务/不发事件，origin 概念删除 |
| R11 | **LOW**：簇 flag 命名不合 REACT_* 惯例、与 grounded 门槛打架 | 已修：`REACT_RESULT_CLUSTER_MIN` + §5.6 割席句 |
| R12 | **PARTIAL 澄清**：observation 非绝对零计数（截断 >20 有「共 N 条」、CN citing 附注） | 已修：§1.2 B1 附注 + §5.5 独立前缀、空候选不加行、既有 digest 单测零破坏 |
| R13 | **PARTIAL**：`resolve_application_number` vs `_resolve_us_app_to_pub_number` 两套反查易被合并抽错 | 已修：§5.2 双反查差异注 |

**三项裁决已落定（2026-09-06，正文相应小节已同步为 v3）**：
1. **预检拒呈现 = 模板引导答复直出，不建任务**（选 B）：chat 主链 INSERT 前拦截、submit 4xx、
   retry 拦同号、detail 格式门——均不产生任务行与失败事件，无统计噪音；"面板重试对格式错无意义"
   的问题随不建任务而消失（worker 执行失败的重试路径保持既有语义）。相关修订：§2.1 G2、§2.3、
   §3、§4、§5.3、§5.4、§6.3、§7、§8、§9、§10、§11、§12 R10；
2. **PCT→WO 实证 spike 提前并行**（选 C）——**已完成（2026-09-06 服务器侧）**：Google XHR 按 PCT 号
   反查**可行**（PCTUS2021059064 → 唯一命中 WO2023075806A1）；直猜恒等命题**证伪**（WO2021059064
   属 ABB 他案）→ §5.2 移除直猜、evidence 删 direct_guess；反查实现排 Phase A 后（flag 默认关）；
   剩余 UAT：WO→EPO docdb kind-less 接受度；
3. **A6 共享门必做 + 细则表按预算 DEFER**：三 examination resolver 前置 `verdict_of()` 一行门计入
   Phase A 必做项；仅各源本地号格式细则表超预算时 DEFER（放行原则兜底），验收口径注明。
