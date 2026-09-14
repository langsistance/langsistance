# 方案：USPTO 取数排序 —— `_score` 强制覆盖导致检索页整体失效

> 批次来源：2026-09-14 承接 `2026-09-13-baiten-quota-frontend-session`「美国噪声第二源」开放项。
> 本文为设计 spec（**只出方案，不含代码改动**），执行拆分沿用单独 plan 流程。
> 基线分支：`feat/baiten-dual-source`（生产分支）。
> 实测工具：`scripts/uspto_sort_probe.py`（本次新增，只读探针）。

---

## 1. 背景与问题

上一轮会话把「美国噪声第二源」记为：自动补跑阶梯的 US 检索式仍返回 20 条全 Provisional，
并猜测「`sort: _score` 不被支持 → 静默回落默认序」。

**该猜测方向相反。** 实测：`sort` 被遵守，而 `_score` 本身在这套语料上是有毒的排序；
API 的默认序（`filingDate desc`）才是好结果。

### 1.1 实测复现（`"LED driver"`，count=1076）

| 排序 | 前 20 条构成 |
|---|---|
| `_score`（**代码强制**） | **20/20 全部 `Provisional Application Expired`**（单号 61207152 / 61170749 / 61005068 / 61528802 / 61670002 … 全为 6x 系临时申请） |
| 省略 `sort`（API 默认） | 18/20 存活（19645023 / 19455116 / 19494266 … 现代在审、授权案） |
| 两者交集 | **0/20** —— 完全不相交的两个结果集 |

`_score` 序在 offset 0 / 20 / 40 三页**全部** 20/20 临时申请。

机理：该端点检索的是**标题级语料**，BM25 的字段长度归一化使「短标题恰好等于检索短语」的记录得分极高。
临时申请标题短而通用（`LED DRIVER` 本身就是完整标题），于是把整页顶满。
这正是 `sources/long_task/candidate_metadata.py:226` 那条 2026-09-01 注释（**RGB** 问题 20/20 Provisional）的成因——
RGB → LED driver 同属这一类。

### 1.2 对上一轮结论的修正

| 上一轮记载 | 实测结论 |
|---|---|
| `dead_filter_diag filtered=20 全 Provisional` 的观察 | **成立**（LED 类确为 20/20） |
| 归因「`sort: _score` 不被支持 → 回落默认序」 | **反向**：sort 被遵守，且默认序更优 |
| 「来自自动补跑阶梯自己的检索式」 | 与阶梯无关；任何走 `_score` 的 US 检索式都中招 |
| 「space-flatten 是噪声源」（上一轮已关） | 关掉是对的，但它不是本条主因 |

（本文作者中途另提过一个假设——日志 `dead_filtered[:5]` 截断导致误读——**该假设也被证伪**，
LED 类确为 20/20，`[:5]` 采样不代表全体但不影响该结论。）

---

## 2. 根因

**代码主动把 API 的排序覆盖为 `_score`，而该语料上 `_score` 的系统性偏向临时申请。**

按影响面排序的三处赋值：

| # | 位置 | 内容 | 是否吃 env 旋钮 |
|---|---|---|---|
| R1 | `sources/agents/react_tools.py:2103` | `_uspto_search_by_query` —— **内置双源检索主路径**，硬编码 `_score` | **否** |
| R2 | `sources/agents/react_tools.py:1029` | `_build_uspto_envelope` —— 动态 KB 工具路径，读 `REACT_USPTO_SORT_FIELD` | 是 |
| R3 | `sources/long_task/recall_sources.py:159` | `fetch_by_numbers` —— 家族号精确取数 | 否（**此处无害**：按号精确匹配，排序不改变命中集合） |

`REACT_USPTO_SORT_FIELD` 默认 `_score`（`react_tools.py:106`）。
**即当前唯一的运维旋钮救不了主路径** —— R1 是硬编码。

引入 `_score` 的原始动机见 `react_tools.py:1025-1028` 注释：工具模板按 assignment 记录日排序（多为噪声），
改用 `_score` 以求「按相关度浮现，不受年代限制」。该动机合理，但在本语料上产生了更严重的新噪声源。

---

## 3. 影响面（实测）

每格 = 前 20 槽位中**通过 dead 过滤**的条数（即真正可展示的候选）。
死掉的是已失效的授权案（`Patent Expired Due to NonPayment`）、`Abandoned`、PCT `PLACED IN STORAGE` 与临时申请。

| 检索式 | count | `_score` 存活 | 默认序存活 |
|---|---|---|---|
| `"LED driver"` | 1076 | **0/20** | 18/20 |
| `"air dryer"` | 283 | 4/20 | 18/20 |
| `"beverage container"` | 3820 | 5/20 | 20/20 |
| `"humidity controller"` | 41 | 6/20 | 10/20 |
| `"semiconductor wafer"` | 3368 | 7/20 | 19/20 |
| `"injection molding"` | 5310 | 19/20 | 18/20 |
| **合计** | | **41/120** | **103/120** |

强制 `_score` 丢掉约 **60%** 可用头部；最坏类（LED/RGB 域）丢掉 **100%** ——
表现为「有命中（`USPTO N hits`）但一条都不可展示」，正是上一轮观察到的现象。

---

## 4. 已实测排除的路径

以下三条**不要**再尝试，均已用探针证伪：

| 路径 | 实测反证 |
|---|---|
| **服务端按类型排除临时申请** | `applicationMetaData.applicationTypeCode:P` 匹配 **0 条**（404 no-match，见 V4）；`… AND NOT …:P` 是**静默空操作**（count 41 → 41，见 V3）。且真实的 63 系临时申请 `63125159` 的 `applicationTypeCode` **是 `UTL` 而非 `P`** —— 类型判据在这套语料上根本不可靠 |
| **靠 `is_provisional_application()` 兜住** | 该谓词读 `type_code`（`candidate_metadata.py:221-229`），而生产双源路径请求的 `RECALL_SEARCH_FIELDS`（`recall_sources.py:30-41`）**不含 `applicationMetaData.applicationTypeCode`**；实测 `fields` 确实生效（请求 10 字段只回 3 个顶层键）→ `type_code` 恒为空串 → **该谓词在生产路径上是死代码**。即便补上该字段也无用（见上条：`:P` 匹配 0 条） |
| **保留 `_score` 只加深分页** | LED 类 offset 0 / 20 / 40 三页**全部** 20/20 临时申请 —— 加深分页对最坏类完全无效 |

**唯一可靠的失效信号是状态串**（`is_dead_status`，`candidate_metadata.py:166-179` 匹配
`expired` / `abandon` / `placed in storage`）—— 现有代码已经在用它，这条要保留。

---

## 5. 方案对比

| 方案 | 机制 | 收益 | 代价/风险 | 改动面 | 可回滚 | 建议 |
|---|---|---|---|---|---|---|
| **A. 接上 env 旋钮** | R1 改读 `REACT_USPTO_SORT_FIELD`，**默认值不变** | 无（行为不变） | 无 | 1 行 | n/a | **前置，建议先做** |
| **B. 默认序改 `filingDate`** | A + 默认值改 `applicationMetaData.filingDate` | 41/120 → 103/120 | 引入**新近度偏差**：窗口变「最新 20 条匹配」而非「最相关 20 条」 | 2 处 | env 一键 | 需产品拍板 |
| **C. 双序并采合并** | 同一 q 发两次（`_score` + `filingDate`），候选求并 | 兼顾相关度与新近度；LED 类可同时拿到默认序那 18 条 | USPTO 请求量 ×2 | 中 | env | 备选 |
| **D. 按需回退（推荐候选）** | 常态仍用 `_score`；**仅当该轮存活候选低于阈值**时用默认序重取一次 | 常态零成本；只在病理类各付一次额外请求 | 需定阈值；阈值过松则频繁双发 | 中 | env | **与 B 二选一** |
| E. 保留 `_score` + 加深分页 | —— | —— | —— | —— | —— | **已排除（§4）** |
| F. 服务端排除临时申请 | —— | —— | —— | —— | —— | **已排除（§4）** |

**关于 B 的一个重要 caveat**：默认序是 `filingDate desc`（新近度），不是相关度。
它「活的多」有一部分是同义反复 —— 新申请还没到过期的时候。
但**下游本来就有语义重排**（`PRESCORE_ENABLED` / `RERANK_ENABLED`，`sources/long_task/semantic_rerank.py`），
取数序只决定**候选窗口**，相关度最终由重排决定。所以 B 的真实代价是：
**候选窗口被新近度筛选**，一篇高度相关的老专利可能进不了窗。
对「现有技术检索 / 自由实施」这类**老专利恰恰是重点**的场景，这是实质损失，需要产品判断。

**关于 D 的阈值**：`"beverage container"` 在 `_score` 下 5/20 存活（非全灭但很差），
`"injection molding"` 19/20（很好）。阈值设在「全灭」只救最坏类；设在「存活 < 10」会覆盖更多类但双发更频繁。
**该阈值是产品决策，不是技术决策。**

---

## 6. 未验证的开放项

1. **6x 系申请号排除**：临时申请单号集中在 60/61/62/63 系。能否用
   `NOT applicationNumberText:61*` 之类做服务端排除？**未验证**（需探针加组；注意 §4 已证明 `NOT` 在类型字段上是空操作，不代表数字前缀也如此）。
2. **USPTO 限流阈值**：方案 C / D 都会增加请求量。当前限流未知，需实测或查官方文档。
3. **`_auto_run_patent_ladder` 复用调用方 `page`**（`react_tools.py:2807`）：每一级阶梯检索式都用**调用方传入的页码**取数，
   而非各自的结果集首页。若 LLM 传了 `page=2`，所有阶梯都读 offset 20。与本条噪声无关，但属同类取数问题，建议一并评估。
4. **`is_provisional_application()` 的去留**：已证实（§4）在生产路径上是死代码且判据不可靠。
   建议**删除或降级为纯辅助**，避免后人误以为临时申请已被该谓词覆盖。相关回归测试在 `tests/test_chat_relevance.py:405-419`，需同步调整。
5. **PCT 记录混入**：`PCTUS2019044798` / `PCTUS2018013228` 以**空 `type_code`** 进入结果，其中部分状态为
   `RO PROCESSING COMPLETED-PLACED IN STORAGE`（dead）。与上一轮「`PCTUS2019044798` 不在交付文件里」的残留症状相关，需单独评估是否过滤。

---

## 7. 验证计划（改动落地后）

1. **回归基线**：改动前跑一次 `--only Y` 记录六条检索式的存活数，作为对照。
2. **复现用例**：`--q '"LED driver"'` 必须从 `_score` 的 0/20 变为 ≥15/20（走默认序或按需回退）。
3. **非退化用例**：`--q '"injection molding"'` 存活数不得低于改动前的 19/20（防止为救最坏类牺牲常态）。
4. **后端回归**：`REDIS_HOST=127.0.0.1 REDIS_PORT=6379 PYTHONUTF8=1 python -m pytest tests/ -q --ignore=tests/test_browser_agent_parsing.py --ignore=tests/test_provider.py`
   —— 基线 **1559 passed / 35 failed / 7 errors**，35/7 为既存环境性失败，须逐条 diff 一致。
5. **单元测试**：排序取值须有测试锁定。
   - 现有 `tests/test_react_tools.py:2676-2689`（`test_envelope_sort_overridden_to_relevance`）**断言 R2 的 sort 恰为 `_score`** ——
     方案 A（默认值不变）该用例仍绿；**方案 B（改默认值）必须同步更新该用例**，否则会红。
   - R1（`_uspto_search_by_query`）目前**无对应断言**，需补。
   - `tests/test_chat_relevance.py:405-419`（`TestProvisionalExclusion`）锁的是 `type_code == "P"` 的排除行为。
     按 §4，该判据在生产路径不可达且语料中 `:P` 匹配 0 条 —— 若采纳开放项 4（删除该谓词），此用例需同步调整。
6. **Python 3.11 语法闸门**：`PYTHONUTF8=1 python -c "import ast;ast.parse(open('文件',encoding='utf-8').read(), feature_version=(3,11))"`。

---

## 8. 附录：探针复现

`scripts/uspto_sort_probe.py`（stdlib，只读，无仓库 import）：

```bash
# 全量：排序对比 / 远距检索式交集 / 头部构成 / 类型约束 / fields 生效性 / 页位构成 / 存活率决策表
USPTO_API_KEY=... python scripts/uspto_sort_probe.py

# 单组
USPTO_API_KEY=... python scripts/uspto_sort_probe.py --only S,X --q '"LED driver"'
USPTO_API_KEY=... python scripts/uspto_sort_probe.py --only Y
```

| 组 | 测什么 |
|---|---|
| S | 同一检索式三种排序（`_score` / `filingDate` / 省略）的返回序是否相同 |
| T | 语义远距的两个检索式结果集是否相交 |
| U | 该检索式头部 20 条的 type/status 直方图 |
| V | 类型能否服务端约束（`applicationTypeCode` 三种写法 + 裸 `:P` 计数） |
| W | `fields` 限制是否生效（决定 `type_code` 是否恒空） |
| X | 同一检索式 offset 0/20/40 三页的 dead / provisional 构成 |
| Y | 六条检索式在 `_score` 与默认序下的存活候选对照表 |

**注意**：本探针请求 `api.uspto.gov`（美国政府公开 API，只读，`limit ≤ 20`），
本机凭据为 `config.ini` / 环境变量 `USPTO_API_KEY`。运行前确认 key 来源，勿把密钥写入仓库。
