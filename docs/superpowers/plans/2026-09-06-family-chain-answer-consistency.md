# 执行 plan：家族深挖链可解析性 + 答复一致性（Phase A）

> 上游设计：`docs/superpowers/specs/2026-09-06-family-chain-answer-consistency-design.md` v3（双路对抗评审 + 三项裁决已落定）。
> 本文档只拆 **Phase A（P1：建议①③④ = 家族深挖链）**；Phase B（需求26）/Phase C（技术簇）另立 plan。
> 分支基线：`feat/baiten-dual-source`（HEAD 7b8fe94，含 spec v3）。执行可在本分支继续或拉 `feat/family-resolve-consistency`。

---

## 0. 执行约定（controller 与子代理共同遵守）

- **TDD 纪律**：每 Task 先写测试（RED，跑出失败）→ 最小实现（GREEN）→ 重构；测试与实现同 commit。
- **环境**：`PYTHONUTF8=1 python -m pytest <file> -q`；pytest 根目录 `E:\online\workspace\copiioai\langsistance`。
- **已知 pre-existing 失败集（勿当回归，与 09-05 一致）**：batch_prompt 12f、memory 5f、session_api 7e、
  searx 3f、patent_analyzer 3f、knowledge_candidates 1f；browser_agent_parsing/provider 收集失败（缺依赖）。
- **回归红线**：
  1. `tests/test_patent_number_parser.py` 全绿（含 `117941643` 9 位 CN 公开号系用例）——任何守卫不得置于 CN 分支前；
  2. `test_react_tools.py` 与 `test_dual_patent_search.py` 中 digest/observation 既有断言零破坏
     （`共 30 条`/`共 50 条`/`_items_digest([]) == ""`）；
  3. 需求#7 测试套（test_session_anchor / test_context_injection / test_task_messages / test_upload_reuse）零回归。
- **代码风格**：不可变模式、early return、函数 <50 行、注释密度随现有文件；禁 console.log/print 调试。
- **commit 纪律**：每 Task 一个原子 commit（conventional commits，中文描述），信息见各 Task。
- **review**：每 Task 完成后由 controller 派 review（重点见各 Task「review 关注点」）；Task 8 全量终审。
- **文件冲突注意**：Task 3/4 动 `api_routes/core.py`+`long_task.py`+`patent_detail.py`，Task 6 动
  `general_agent.py`+`react_tools.py`，Task 5 动 `celery_worker.py`+`status_manager.py`，Task 7 动三个
  examination 模块——**不同文件集，可并行子代理**；Task 3 先行（产出共享 `failure_guidance`）。

---

## 1. 任务总览与依赖

| # | 任务 | 对应 spec | 主要文件 | 依赖 |
|---|---|---|---|---|
| T1 | parser：pct/wo 分支 + PCT 头剥离 + ≥9 位守卫 | §5.1 | patent_number_parser.py | — |
| T2 | translator：TranslateResult/translate/verdict_of + `_resolve_us_app_to_pub_number` 收敛 | §5.2 | patent_id_translator.py(新)/celery_worker.py(仅抽函数) | T1 |
| T3 | `failure_guidance()`/reason_code + chat 主链预检门（引导答复直出，INSERT 前） | §5.3/§5.4/§6.3 | status_manager.py(或 task_messages.py)/core.py | T2 |
| T4 | submit/retry 预检门 + detail 路由轻量门 | §5.3 入口 2-4 | long_task.py/patent_detail.py | T3 |
| T5 | worker 失败单点出口（外层 notify+MySQL 成对）+ families Phase0 候选改调 translator | §5.4 | celery_worker.py/status_manager.py | T3 |
| T6 | 家族意图预路由窄门 + `_dispatch_from_mysql` task_type 分派（A9） | §5.5b/§2.2 | general_agent.py/react_tools.py/long_task.py | T1 |
| T7 | A6 共享门（三 examination resolver 前置 verdict_of） | §5.3 A6 | china/epo/japan_examination.py | T2 |
| T8 | 全量回归对照 + 终审（opus/controller-run） | §2.3 自动化部分 | — | 全部 |
| S1 | 服务器侧 spike（PCT→WO 实证）——**已完成 2026-09-06**；剩 WO→EPO kind-less UAT | §10/§11 | 服务器 | 部署时补 UAT |

拓扑序：T1 → T2 → T3 → {T4 ∥ T5 ∥ T6 ∥ T7} → T8。T3 是共享引导层的唯一前置。

---

## 2. Task 1：号码解析扩展（parser pct/wo + 守卫）

**目标**：`parse_patent_identifiers` 认识 PCT/WO 形态、≥9 位裸数字不再产 US ambiguous；
`format_number_guidance` 输出 pct/wo/unsupported 引导。现有 CN/US 行为逐字节兼容。

**规格要点（spec §5.1）**：
1. token 扫描先匹配 `PCT[/\s]?{RO}[/\s]?{YYYY}[/\s]?{NNNNNN}`（RO 两字母，年份 4 位 + 序号 ≥5 位），
   剥离 PCT 头后统一校验，**防 `PCT US2021 059064` 落 US prefix**；
2. `WO` 前缀从 `_external()` 提升：`WO{YYYY}[/\s]?{NNNNNN}([A-Z]\d*)?` → id_type=wo，
   display `WO{YYYY}/{NNNNNN}`，lookups `["WO{YYYY}{NNNNNN}"]`；
3. `_classify_bare`：**保留既有 CN 分支在前**（12/13 位 19|20 开头 CN 申请号、9 位 `1` 开头 CN 公开号），
   其后新增：≥9 位其余裸数字 → `id_type="unsupported"`（reason 中文：长度不符美国号段，疑似残缺国际申请号/
   含 PCT/WO 前缀），lookups=[]，country=""；
4. pct 输出：`id_type="pct"`、`display="PCT/{RO}/{YYYY}/{NNNNNN}"`、`country="WO"`、`meta={"office": RO}`、
   `lookups=[]`、confidence high（显式前缀）或 medium（空格/斜杠变体）；
5. `format_number_guidance` 接受 pct/wo/unsupported 并输出引导句（zh/en 双语文案，通用不固化提问词）；
6. `decide_number_source`：pct/wo → 不强制单源（None）。

**TDD 断言清单（先写 `tests/test_patent_number_parser.py` 内新增用例 → RED）**：
- `PCTUS2021059064` → pct / display `PCT/US2021/059064` / country WO / meta.office US / confidence high；
- `PCT/US2021/059064`、`PCT US2021 059064` → 同上 display（形态归一）；
- `PCTCN202312345678`/`PCT/CN2023/12345678` → pct / office CN；
- `WO2021059064`、`WO2021/059064`、`WO2021/059064A1` → wo / display `WO2021/059064` / lookups 含 `WO2021059064`；
- 裸 `2021059064`、`12345678901` → unsupported（非 US ambiguous）；lookups 空；
- **回归红线**：`117941643` → CN pub（既有用例不动）；12/13 位 CN、6-8 位 US ambiguous、CN 前缀/US 前缀
  既有用例全绿；
- guidance：含 pct 候选 → 输出含 `PCT/` 规范读法引导；含 unsupported → 输出补全前缀建议；
  country 非 CN/US 旧行为（EP/WO unsupported 低置信路径）仅在新 wo 分支上变更。

**改动文件**：`sources/patent_number_parser.py`、`tests/test_patent_number_parser.py`。
**review 关注点**：分支顺序（CN 在守卫前）、形态正则不过度（勿误吞 `US2021/059064` 类真实美国文献号——
   仅当 query 上下文含 PCT 或显式 PCT 头才判 pct）、guidance 文案通用性、无魔法数字。
**commit**：`feat: patent_number_parser 支持 PCT/WO 国际申请号 + ≥9 位裸数字守卫（需求26批次 A1）`

---

## 3. Task 2：翻译层 translator + 可解析性判定

**目标**：新建 `sources/patent_id_translator.py`，产出 §5.2 判定表语义；把 celery_worker 的
`_resolve_us_app_to_pub_number`（1815-1920）收敛为可复用实现（纯重构先行，行为不变）。

**规格要点（spec §5.2）**：
1. `TranslateResult{patent_id, verdict: "resolvable"|"unresolvable", candidates{epo_docdb[], uspto[], cnipa[]},
   evidence: "local"|"direct_guess"|"reverse_lookup"|"", reason, confidence}`；
2. `translate(patent_id, scenario="", reverse_lookup=False)`：按归类表产 candidates+verdict——
   - US grant/pub（≤8 位或带 US 前缀/kind）→ resolvable（epo 候选经收敛的 `_resolve_us_app_to_pub_number`；
     evidence=local）；
   - US 申请号 → resolvable（uspto 候选原样；epo 候选缺 grant 时 confidence=low）；
   - CN 申请/公开号 → resolvable（cnipa 候选归一），交 regional executor 自管；
   - WO 公开号（id_type=wo，**用户直接提供**）→ resolvable，evidence=local
     （docdb 候选 `WO{YYYY}{NNNNNN}`，kind 可省性由 UAT 定，不阻塞本任务）；
   - PCT（id_type=pct）→ reverse_lookup=False 时 **unresolvable**（reason 引导文案）；
     =True 时走反查通道（本任务留接口；通道 spike 已证可行——PCTUS2021059064 → WO2023075806A1
     唯一命中，2026-09-06 服务器实证；实现排 Phase A 后，未实现时按 unresolvable fail-closed）；
     **不构造 WO 直猜候选**（恒等命题已证伪：WO2021059064 属 ABB 他案）；
   - 未知前缀/unsupported → unresolvable；
   - 任何内部异常 → 上抛（上层放行原则）；
3. `verdict_of(patent_id, scenario)`：同步轻量版，无网络路径（仅本地正则 + parser 结果），供
   detail 路由/预检非 async 段；内部只调 parser 与正则，不触发任何反查；
4. **收敛重构**：`celery_worker.py` families Phase 0 候选构造（1937-1963）改调 translator
   （本任务只做"抽取+同行为"，Phase 0 的 fail 出口改造留给 T5）；
   注意与 `uspto_download.resolve_application_number` 是两套反查——**不得合并**。

**TDD 断言清单（`tests/test_patent_id_translator.py` 新建 → RED）**：
- 判定表逐行（mock `_resolve_us_app_to_pub_number` 返回/异常两种）：
  US grant→resolvable；US 申请号→resolvable(low)；CN→resolvable；WO→resolvable(direct_guess)；
  PCT(reverse_lookup=False)→unresolvable 且 reason 含「WO 公开号/国家阶段申请号」；unsupported→unresolvable；
- reverse_lookup=True 且实验通道未实现 → 同 unresolvable（fail-closed，不抛裸异常）；
- `verdict_of` 无网络：对同一批输入与 translate 结论一致（局部样本）；
- translator 内部抛异常 → 异常上抛（由上层放行测试覆盖）。
- 收敛重构回归：families Phase 0 候选构造行为不变（现 celery_worker 相关既有测试若存在则全绿）。

**改动文件**：`sources/patent_id_translator.py`(新)、`sources/patent_number_parser.py`(仅引用)、
`celery_worker.py`(抽函数重构)、`tests/test_patent_id_translator.py`(新)。
**review 关注点**：verdict 只拦"确定性不可解析"；evidence 语义清晰；无网络副作用的 verdict_of 纯净性；
双反查不混淆；fail-closed 不裸抛。
**commit**：`feat: 专利号翻译层 translator/verdict + US 反查收敛（需求26批次 A2）`

---

## 4. Task 3：共享引导模板 + chat 主链预检门（引导答复直出）

**目标**：预检拒在 chat 主链以**模板引导答复**收尾——不建任务行、不派 celery、无失败事件（裁决①）。
worker 侧失败消息（T5）复用同一 `failure_guidance`。

**规格要点（spec §5.3 入口1 / §5.4 / §6.3）**：
1. `failure_guidance(task_type, reason_code, error="") -> str` 纯函数（放 `status_manager.py` 或
   `task_messages.py`，T3 定稿后 T5 引用）；reason_code 枚举：`ERR_UNRESOLVABLE_ID`（verdict 直接产码）、
   `ERR_EPO_REMOTE`（worker 错误含 5xx/timeout/credentials 归类）、`ERR_OTHER`；
   文案 = 通用引导段（spec §5.4 模板表），zh/en 双语文案（lang 参数），**不含任何用户提问词**；
2. chat 主链预检门：`core.py` scenario 判定（1290 区）后、会话/任务 INSERT（1376）**前**——
   仅 `scenario=="families"`；`verdict_of(patent_ids[0], "families")` 为 unresolvable →
   走**引导直答出口**：以 assistant 文本（`failure_guidance(...)` + 简短前缀"未能启动该分析任务：
   号码无法解析"）结束本轮（复用 chat_fallback 的降级出口形态：SSE 直出文本 + end；
   不创建任务、不 INSERT、不 delay、不 track long_task:submit/fail）；
3. resolvable → 原流程不变（translator 产物存 context 变量供 T5 阶段注入 celery_params 候选，
   本任务不接线）。
4. 现有 chat_fallback（1290-1333）不动。

**TDD 断言清单（`tests/test_long_task_precheck.py` 新建 + core 集成测试）**：
- `failure_guidance` 三 reason_code 各自文案含对应建议；zh/en；不抛异常（error 超长截断）；
- chat 主链 mock：families + `PCTUS2021059064`（mock verdict_of → unresolvable）→ 断言
  **INSERT 未执行（任务行不存在）、`execute_family_analysis.delay` 未调用、SSE 收尾为引导文本、
  analytics 无 long_task:submit/fail 事件**；
- 同输入 resolvable（US grant）→ 原流程照常（INSERT+delay 调用，回归既有路径测试）；
- translator 异常（mock 抛）→ 放行：INSERT+delay 照常（旧行为回归）。

**改动文件**：`sources/long_task/status_manager.py`（或 task_messages.py）、`api_routes/core.py`、
`tests/test_long_task_precheck.py`(新)。
**review 关注点**：INSERT 前拦截点不破坏既有 created 消息/SSE 时序（created 消息在 INSERT 后——
   确认拦截发生在任何任务相关事件之前）；引导答复与普通答复在会话中的落库路径一致（_store_current_turn）；
   无失败事件但用户仍看到回复（UX 完整）；chat_fallback 不被误触发。
**commit**：`feat: 家族任务提交前可解析性预检——不可解析号引导直答（需求26批次 A3）`

---

## 5. Task 4：submit/retry 预检门 + detail 路由轻量门

**目标**：预检门覆盖其余三入口（裁决①的呈现规则）。

**规格要点（spec §5.3 入口 2-4）**：
1. `submit_long_task`：`_normalize_submit_patent_id`（128-160）后、`delay`（300-308）前加
   `verdict_of` 门：unresolvable → HTTP 4xx（422 带结构化 body：`{error, code:"ERR_UNRESOLVABLE_ID",
   guidance: failure_guidance(...)}`），**不建任务行**；弱正则保留作 fallback 校验；
2. `retry_long_task`（338-417）：`_dispatch_retry_task` 前同门：unresolvable → 4xx + guidance，
   任务保持原 worker 失败态（不放行同号重放）；
3. detail 路由（`patent_detail.py` `_fetch_claims` 648-651 / `_fetch_spec_pdf` 596-601）调
   `resolve_application_number` 前：`verdict_of(id)` unresolvable（pct/wo/unsupported/外文形态）
   → 返回引导错误（422 + guidance 文案），**不剥前缀送 resolve**；US/CN 合法形态零影响。
4. 四入口共享 `verdict_of`，无网络副作用。

**TDD 断言清单（并入 `tests/test_long_task_precheck.py`）**：
- submit：PCT/unsupported 号 → 422 + guidance + **任务行未创建** + delay 未调用；
  US grant → 200 流程照常；
- retry：原 failed 任务 + unresolvable 号 → 拒绝（4xx），任务状态不变；
- detail：`/patent/uspto/.../claims` 带 `WO2021059064`/`PCTUS2021059064`/裸 10 位 →
  422 + guidance，**resolve_application_number 未被调用（mock 断言）**；
  带合法 US 号 → 与现状一致（mock resolve 正常调用）；
- detail 门对 CN source 号零影响（CN 走佰腾分支不被误拦）。

**改动文件**：`api_routes/long_task.py`、`api_routes/patent_detail.py`、`tests/test_long_task_precheck.py`。
**review 关注点**：4xx 语义与前端既有错误处理兼容（找前端错误呈现约定）；CN 分支不被误伤；
retry 语义不破坏 worker 瞬时失败的重试（仅拦 unresolvable）。
**commit**：`feat: submit/retry/detail 三入口可解析性门 + 引导错误（需求26批次 A3）`

---

## 6. Task 5：worker 失败单点出口 + Phase 0 候选接线

**目标**：families 失败事件单发、MySQL 终态与 Redis 一致；Phase 0 EPO 候选改调 translator。

**规格要点（spec §5.4）**：
1. `celery_worker.py` families Phase 0 fail 分支（1980-1988）：删除内联 `set_task_failed` +
   `_update_mysql_progress('failed', 0)`，仅 `return {'status':'failed','task_id','error'}`；
2. 外层 returned-failed 分支（2954-2958）升级：`_notify_terminal_failure(task_id, error)` **后追加**
   `_update_mysql_progress(task_id, 'failed', 0)`（与同文件其余 executor 异常分支惯例一致，
   见 2972-2973/3807/3863 附近成对模式）——Redis failed + MySQL failed 各一次；
3. `_notify_terminal_failure` 内容改拼 `failure_guidance(task_type, reason_code, error)`：
   families/executor 错误按 reason_code 归类（5xx/timeout/credentials → ERR_EPO_REMOTE；
   InvalidCountryCode/全格式 404 → ERR_UNRESOLVABLE_ID；其余 ERR_OTHER）；task_type 从
   task 状态/executor 名推导；
4. families Phase 0 候选构造（1937-1963）改调 `translator.translate(patent_id, scenario="families")`
   的 `candidates.epo_docdb`（T2 收敛后同函数）——**仅替换候选来源，不改变尝试循环语义**；
5. 幂等：executor 不再写 failed → `long_task:fail` 单发（不做相等去重；不引入并发窗口条款）。

**TDD 断言清单（`tests/test_failure_terminal_state.py` 新建 → RED）**：
- mock：executor 返回 failed dict → `set_task_failed` 恰 1 次、`_update_mysql_progress('failed')` 恰 1 次、
  analytics `long_task:fail` 恰 1 次；
- 失败消息 content 含 reason_code 对应引导段（InvalidCountryCode 案例 → WO 公开号建议）；
- families Phase 0 候选：mock translator 返回 candidates → 尝试顺序 = translator 候选序；
  translator 异常 → 候选回退 `[patent_id]`（旧行为保底）；
- 其余 executor（china/epo/japan/prosecution）既有失败语义零回归（各自套件）。

**改动文件**：`celery_worker.py`、`sources/long_task/status_manager.py`、
`tests/test_failure_terminal_state.py`(新)。
**review 关注点**：MySQL 终态与 resume/面板依赖一致；notify 只在一处发 analytics；Phase 0 候选回退
  不引入新失败面；reason_code 归类不做子串匹配（结构化）。
**commit**：`fix: 家族任务失败单点出口——analytics 单发 + MySQL 终态成对（需求26批次 A4）`

---

## 7. Task 6：家族意图预路由窄门 + resume 错 executor 修复

**目标**：KB 无家族项用户的「pct/wo 候选 + 家族分析意图」请求不再经 ~16s 空转（spec §5.5b）；
A9 `_dispatch_from_mysql` 按 task_type 分派。

**规格要点**：
1. 预路由窄门（`general_agent.py` 2061-2079 预路由区块，`_match_long_task_intent` 过滤之后）：
   - 条件 A：parser 产出 pct/wo/unsupported 候选（`self._number_candidates`）；
   - 条件 B：家族**分析**意图——复用 `react_tools._FAMILY_INTENT_KEYWORDS_ZH`（355-357）
     与审查分析动词（审查/授权/驳回/差异/分析…）交集命中 prompt；**纯检索动词（检索/查找/搜索）+ 家族词
     不命中**（防误路由）；
   - 条件 C：registry 无 type-3 KB 家族项命中（既有路径优先，不抢占）；
   - 三条件齐 → 直接 `_build_long_task_intent(...)`（families）——绕过 ReAct 空转；
2. 窄门命中后 core 分类/预检照常（T3 门兜底 unresolvable）；
3. `_dispatch_from_mysql`（`long_task.py` 19-55）：按 MySQL task_type 分派 executor
   （family_analysis→execute_family_analysis；其余→execute_patent_analysis），消除 resume 错配。

**TDD 断言清单（`tests/test_family_preroute.py` 新建 + 既有 matcher 测试扩展）**：
- pct 候选 + 「分析 X 及其全球同族申请的审查差异」→ 预路由命中 families（不落 ReAct）；
- 同 pct 候选 + 「检索与 X 同族且美国授权的专利」→ **不命中**（走检索）；
- WO 候选 + 家族分析词 → 命中；裸 US 号 + 家族分析词（KB 无项）→ 不命中窄门（维持现状——KB 场景
  覆盖由既有 type-3 路径处理，本任务不扩大）；
- KB 有 type-3 家族项 → 既有 `_match_long_task_intent` 路径优先（窄门不抢占）；
- `_dispatch_from_mysql`：family_analysis 行 → 分派 execute_family_analysis（mock delay 断言）；
  批量行 → execute_patent_analysis（现状回归）。

**改动文件**：`sources/agents/general_agent.py`、`sources/agents/react_tools.py`(词表/常量引用)、
`api_routes/long_task.py`、`tests/test_family_preroute.py`(新)。
**review 关注点**：窄门条件不破坏 knowledge=None 防呆护栏（spec §5.5b——不放开通用 deep 项）；
误路由表驱动反例齐全；词表复用不新增硬编码提问词。
**commit**：`feat: 家族意图确定性预路由窄门 + resume 按 task_type 分派（需求26批次 A5）`

---

## 8. Task 7：A6 共享门（三 examination resolver）

**目标**：CN/EP/JP 三个 resolver 委托 `lookup_family` 前加 `verdict_of()` 一行门（裁决③：共享门必做；
细则表按预算 DEFER）。

**规格要点（spec §5.3 A6）**：
- `china_examination.resolve_cn_application_number`（141-223）、`epo_examination.resolve_ep_application_number`
  （89-172）、`japan_examination.resolve_jp_application_number`（56-135）各自 `lookup_family` 调用前：
  本地号形态正则命中（各自既有 is_* 判定）→ 照旧；未命中（外文/裸号）→ `verdict_of(patent_id)`：
  unresolvable → 抛/返回结构化错误（由调用方经既有 set_task_failed 通道带 guidance——本任务只保证
  reason_code 传入，T5 模板后即有引导文案），**不再白送 EPO**；
- resolvable → 原 lookup_family 流程不变；
- **细则表 DEFER**：不新增 JP 4 段式/EP 消歧细则（放行原则兜底）。

**TDD 断言清单（扩展三个 examination 既有测试或新建 `tests/test_exam_resolver_gate.py`）**：
- 三个 resolver × 本地号 → 照旧直归（既有用例零回归）；
- 三个 resolver × PCT/unsupported 号（mock lookup_family 不应被调用）→ 返回/抛引导性错误；
- 三个 resolver × translator 异常 → 放行 lookup_family 旧行为。

**改动文件**：`sources/long_task/{china,epo,japan}_examination.py`、测试。
**review 关注点**：各 resolver 现有错误通道兼容（不引入新异常类型破坏调用方）；门只拦确定性形态。
**commit**：`feat: CN/EP/JP examination resolver 前置可解析性共享门（需求26批次 A6）`

---

## 9. Task 8：全量回归对照 + 终审（controller-run）

**步骤**：
1. 逐 Task 回归：本 plan 新增/扩展测试文件全绿（T1-T7 各自文件）；
2. 全量：`PYTHONUTF8=1 python -m pytest <相关套件> -q`，输出须等于「基线 + 本批次新增」且
   新增失败 = 0；对照 pre-existing 失败画像（§0 清单）无漂移（worktree 对照法同 09-05：需拷
   gitignored config.ini/.env；混跑 collection error 为共享状态干扰，逐文件单独跑）；
3. 终审（opus，review-package）：闭合回路逐链核对 spec §2.3 自动化验收 1-6；
4. 产出终审意见（Ready to merge: Yes/No + 残留清单）。

**commit**：无（或终审修复随对应 Task 补丁）。

---

## 10. S1：服务器侧 spike（已完成）+ 剩余 UAT（非代码任务）

**Spike 已完成（2026-09-06 服务器侧 curl），实证结论**：
1. **Google 反查通道可行**：`https://patents.google.com/xhr/query?url=q%3DPCTUS2021059064` →
   唯一命中 `WO2023075806A1`（Rakuten Mobile 异常检测/根因案；filing_date 2021-11-12 与申请号吻合；
   JSON 直接含 publication_number + family_metadata，比页面抓取稳定）——反查实现以 XHR 端点为参考；
2. **WO 直猜恒等命题证伪**：`WO2021059064`（直猜号）命中的是 ABB 电动车充电案
   （priority 2019-09-23 / 公布 2021-04-01），与 PCT/US2021/059064 无关；真实公开号
   WO2023075806A1（2023-05-04 公布）→ translator **不再构造直猜候选**；
3. 结论已回写 spec §5.2/§2.3/§11（evidence 语义、无直猜）。

**剩余 UAT（部署窗口，服务器有 EPO 凭据时）**：真实 WO 号送 EPO OPS docdb 的 **kind-less 接受度**——
试发 `WO2023075806` vs `WO2023075806A1` 对比；结论回写 spec（影响 docdb 候选是否带 kind）。

**部署 UAT（spec §2.3 轨迹复跑清单全项）**：推送分支 → /opt/langsistance 部署 → 重启 celery/uvicorn →
复跑 09-04 轨迹核对 8 项日志信号（PCT 识别、无 16s 空转、引导直答、单发 fail、MySQL 终态、
detail 门、文案一致性、簇标注[Phase C 未上,跳过]）。

---

## 11. 风险与兜底

- T3 拦截点若与 created 消息时序纠缠 → 拦截点前移（scenario 判定后立即），created 事件只对
  真实创建的任务发；
- verdict 误判（如 10 位号段未来被 USPTO 采用）→ 放行原则：细则 DEFER 期间误拦仅表现为引导答复，
  跟踪答复转成功比例；
- translator 收敛重构引入 celery_worker import 环 → translator 独立模块不 import celery_worker
  （`_resolve_us_app_to_pub_number` 以函数参数注入或抽到 translator 可引用的公共位置）；
- Google 反查通道：**spike 已服务器实证可行**（PCTUS2021059064 → WO2023075806A1 唯一命中）；
  实现排 Phase A 后；EPO kind-less 接受度待服务器 UAT（不阻塞代码合入）。

---

## 12. 交付物清单

- 代码：T1-T7 各原子 commit（7 个）+ T8 终审记录；
- 测试：新增 `test_patent_id_translator.py` / `test_long_task_precheck.py` / `test_failure_terminal_state.py` /
  `test_family_preroute.py`（及扩展文件），全绿；
- 文档：本 plan 与 spec v3 已跟踪；S1 结论回写 spec；
- 需求列表回填：xlsx 需求 6/22/26 证据状态（部署验收通过后由分析账本会话执行）。
