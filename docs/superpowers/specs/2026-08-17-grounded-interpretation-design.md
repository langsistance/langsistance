# 检索后接地解读：技术维度贯穿的闭环设计

**日期:** 2026-08-17
**项目:** langsistance (CopiioAI 后端) — US 专利关键词检索链路
**状态:** 已批准（用户确认后写入）

## 1. 背景与差距分析

对 langsistance 的 US 专利关键词检索链路做系统性质量改造，目标追平 workbuddy+智慧芽报告的检索质量。上一会话（2026-08-16/17）完成了预检索技术解读组件（gpt-5.6-terra）、家族 flash 评分机制、解读产业地图（main_lines/key_players），判据专利 12289808 首次进入最终列表。

**当前缺口（解读质量 71 vs 智慧芽 95.5，主项 = 主线/玩家层）：**

1. **信息源问题**：智慧芽解读的玩家/主线来自**真实检索结果**（"检索发现该领域存在两条主线"），我们目前是纯预检索 LLM 知识。玩家细分度错层（terra 给照明巨头 Signify/OSRAM/Lumileds/TI/ADI，而领域专精玩家是 ERP Power），第二条主线（显示/背光 RGB 色温控制，三星/LG/长虹/中电熊猫，DAC/PWM+MOS）完全缺失。
2. **元件级深度**：即使 prompt 已给 VCR 作为示例词（"如 error amplifier、voltage controlled resistor…"），terra 仍不产出 VCR——纯 LLM 知识连不上该领域，光改 prompt 无效。
3. **维度结构缺失**：智慧芽解读是**技术维度分解**（核心器件层/控制算法电路层/场景应用层，各带标签），我们目前是扁平的 scheme+main_lines。
4. **打分并发不足**：flash 评分 50 条 45-136s（批 25、50 条仅 2 路并行）；语义模型 embedding 是同步调用直接阻塞事件循环，单次全池大 batch。

## 2. 设计目标

1. 生成智慧芽水平的专利检索级技术解读：**本质句 + 技术维度骨架（每维度带标签/词汇/查询）+ 每维度主线（带代表申请人 + 实现描述）+ 数据驱动专精玩家**。
2. 解读贯穿全链路：预检索阶段产出维度指导查询，检索后从候选池数据接地生成主线/玩家，闭环反哺后续检索（补查询/补 CPC/评分加权）。
3. 增强打分并发能力（flash 与语义模型两侧）。
4. 全部修改通用，不针对任何测试提问（铁律，第三次重申）。

## 3. 架构总览

```
[预检索] terra 解读（升级）→ dimensions(2-3, 三层骨架) + scheme/structure_terms + 按维度查询
    ↓ 查询按维度进 ladder（每维度 1 条）
[检索] USPTO 检索 → 候选池（metadata: title/applicant/cpc/family/评分）
    ↓ 已评分 ≥15 触发（单次标志）
[接地] flash 合成（新模块）→ 按维度组织的主线/玩家/补查询/补CPC（失败降级纯统计）
    ↓ 反哺
[闭环] ① 补查询自动执行并入池  ② 补 CPC 进 recall expansion  ③ rubric 升级（玩家榜真信号）
[并发] flash 批 10 + 信号量 6；语义 embedding 分块 64 + 线程池并行
```

## 4. 详细设计

### 4.1 预检索解读升级（terra，`technical_interpretation.py`）

`dimensions` 成为新主输出：

```json
{
  "dimensions": [
    {"name": "...", "role": "核心器件/电路层",
     "terms": ["..."], "queries": ["..."]},
    {"name": "...", "role": "控制算法/电路层", ...},
    {"name": "...", "role": "场景应用层", ...}
  ],
  "scheme": "...", "structure_terms": [...], "independence_terms": [...]
}
```

- **固定三层通用骨架**：核心器件/电路层（实现功能的硬件核心）、控制算法/电路层（实现功能的控制逻辑）、场景应用层（功能落地的应用与接口）。
- 模型规则（全泛化措辞）：按纵深拆解为技术维度，默认三层；某层在目标领域无实质内容时并入最接近的层并说明；**输出 2-3 个，禁止超过 3 个**。
- **parse 硬控制**（不依赖模型自觉）：`dimensions[:3]` 截断 + 角色标签去重 + 空维度丢弃。
- 每维度输出：维度名、维度方案词（英文）、1-2 条该维度布尔查询。
- `main_lines` 保留但降级为接地合成的种子（权威主线在检索后由数据产出）。
- **`merge_interpretation_queries` 改按维度取查询**：每维度 1 条进 ladder 头（≤3 槽，MAX_INTERP_LADDER_SLOTS），保证检索覆盖所有层面而非仅表面词最像的维度。
- scheme/structure_terms/independence_terms/scenarios 保留现有行为。

### 4.2 接地合成模块（新文件 `sources/long_task/grounded_interpretation.py`）

三个纯函数 + 一个 async 入口，**任何路径绝不抛异常**：

- `candidate_stats(candidates)` — 申请人频次、CPC 频次（纯代码，零 LLM；任何失败此层仍可用）。
- `build_synthesis_input(...)` — 组装 flash 输入（prompt 全泛化，铁律审计项）。
- `merge_grounded(stats, llm_out, pre_interpretation)` — 合并：flash 失败 → **纯统计版接地解读**（players=全局频次榜、lines=CPC 标题分组）；LLM 输出永远可缺失。
- `synthesize_grounded(candidates, pre_interp, cpc_hints)` — async 入口，flash 调用，超时/失败 → 统计版保底 → 再失败 → None。

**输入**（纯确定性数据，无摘要下载）：已评分候选 top-30（title/applicant/cpc_codes/家族/评分）+ 申请人频次统计 + CPC 频次统计 + 预解读维度骨架。

**输出**（按维度组织）：

```json
{
  "dimensions": [
    {"name": "...", "role": "...", "line": "主线描述",
     "representatives": ["..."], "players": [...], "cpc": [...]}
  ],
  "supplementary_queries": [...],
  "supplementary_cpc": [...]
}
```

- flash 以预解读维度为初始骨架，依据候选数据可调整/合并/拆分维度。
- 模型：`REACT_GROUNDED_MODEL`（默认与评分同一 flash provider）。

### 4.3 触发点与反哺（`react_tools.py`）

新增 `_grounded_synthesis_round(agent, entry, lang)`，在 `execute_action` 主搜索路径的 `_auto_second_round` 之后、`_auto_feedback_round`/`_recall_expansion_round` 之前调用：

- **触发**：`agent._grounded_done` 单次标志（create_agent 按请求重置）+ 已评分候选 ≥15（`REACT_GROUNDED_MIN`）；不足 → 跳过并置位，预解读继续引导。
- **探针日志（铁律）**：`grounded_interpretation probe — pool=N scored=M trigger=T` 每请求必打；成功后打 `grounded_interpretation — lines=... players=...`。
- **反哺① 补查询**：`supplementary_queries` 立即自动执行（复用 `_invoke_and_merge`，同 `_auto_feedback_round` 机制），并入池并评分。
- **反哺② 补 CPC**：追加 `agent._grounded_cpc`，`_recall_expansion_round` 的 fetch_by_cpc codes = cpc_hints + grounded_cpc（受 RECALL_MAX_CPC 上限约束）。
- **反哺③ 评分/家族**：后续 `_rank_pending_pool` 的 rubric 用 `format_interpretation_rubric(merged)`（接地优先，预解读的 scheme/structure_terms 保留）。

### 4.4 评分加权与家族联动 + 错误处理

- **rubric 升级**：players 由"仅背景参考、不构成相关性依据"改为**真实玩家榜（数据驱动）**——"申请人命中该榜且其他信号吻合时，视为同领域证据，评分可上调 3-5 分（满分 100）"。由评分 LLM 判定，不做代码级硬加分（防单信号误判）。
- **家族联动零新代码**：设计内预期路径——同族成员因申请人命中玩家榜被 rubric 抬升 → 越过 REACT_FAMILY_SEED_MIN → 成为种子 → 家族机制自然抬同族。
- **错误处理**：合成任何失败 → 统计版保底 → 再失败 → None，流程不变；flash 挂不影响预解读和主循环。

### 4.5 打分并发增强

**Flash 侧（`chat_relevance.py`）**：
- `SCORE_BATCH_SIZE` 25 → 10（env 可调）：50 条 → 5 路并行。
- 新增 `SCORE_MAX_CONCURRENCY`（默认 6，`asyncio.Semaphore`）：并行上限，防网关 429 限流。
- `score_candidates_concurrent` 用 `asyncio.gather` + `Semaphore` 重写；家族评分、补查询并入池共用同一函数自动受益。

**语义模型侧（`semantic_rerank.py`）**：
- `semantic_scores_batch` / `rerank_candidates` 改 **async**：embedding 调用走 `asyncio.to_thread`（不再阻塞事件循环）。
- **分块并发**：每块 `SEMANTIC_BATCH_SIZE=64` 条，块间 `asyncio.gather` 并行。
- 失败降级不变（→ {} 或原序返回）；调用方 `react_tools.py` 两处补 `await`。
- 纯数学函数（cosine_similarity / fuse_ranking / 归一化）保持同步不变。
- 单元测试同步更新（签名变化）。

### 4.6 验证方案

**单元测试**（`tests/test_grounded_interpretation.py` 新增 + 既有测试更新）：
- 统计函数（申请人/CPC 频次）用 fixture 候选。
- prompt 组装通用性审计：断言分层措辞为通用角色（核心器件/控制算法/场景应用），测试提问词汇零出现。
- flash 失败降级统计版；与预解读合并；单次触发标志（二次评分不再触发）。
- 补查询与已有 ladder 去重。
- parse 硬控制：维度 >3 截断、角色去重、空维度丢弃。
- `score_candidates_concurrent` 并发上限行为；`semantic_scores_batch`/`rerank_candidates` 签名更新。

**生产验证**（服务器部署后跑测试问题）：
1. 日志 `grounded_interpretation — lines=... players=...` 出现。
2. **players 数据驱动**：核心器件维度出现专精玩家（该测试问题预期 ERP Power 从数据浮现）。
3. 主线 ≥2 条，第二条（显示/背光向）从数据出现。
4. 补查询自动执行、补 CPC 进 recall expansion（日志确认）。
5. 最终列表结构判据：本质句 + 维度骨架（2-3 个，各带角色标签，期望三层齐全）+ 每维度主线带代表申请人 + 玩家数据驱动；rubric 方案层锚点不倒退（≥17/20）。
6. **并发收益**：`relevance scoring — elapsed=` 50 条 45-136s → ~10-25s；`semantic prescore — elapsed=` 正常且不再卡事件循环。

**Env 旋钮**（沿用现有模式）：
- `REACT_GROUNDED_ENABLED=1`（默认）、`REACT_GROUNDED_MODEL`（默认评分同款 flash）、`REACT_GROUNDED_HEAD=30`、`REACT_GROUNDED_MIN=15`
- `SCORE_BATCH_SIZE=10`、`SCORE_MAX_CONCURRENCY=6`、`SEMANTIC_BATCH_SIZE=64`

## 5. 非目标（YAGNI）

- 用户可见的"技术主题解读"展示块（用户明确未选）。
- 两次渐进式重合成（recall 后再合成一次）——单次触发先行，数据不足再评估。
- 12289808 稳定进 top-100 不作为本次硬判据（跨轮方差是另一层问题）。
- 摘要下载（候选元数据无摘要，主线聚类靠 title/applicant/CPC 组合，已验证足够）。

## 6. 通用性铁律（验收项，第三次重申）

- 分层措辞只用通用角色名（核心器件/控制算法/场景应用），不含任何测试提问词汇。
- 维度示例用其他领域措辞（沿用温控闭环/电机变频风格）。
- 提交前 grep 审计：prompt/代码中测试提问词零出现（纳入单元测试）。
- 测试问题只出现在验证运行里，永不进代码和 prompt。

## 7. 部署与验证流程

1. 实现 → 单元测试全绿 → 通用性审计 grep。
2. 提交 → 同步服务器 → 重启。
3. 跑测试问题生产轮 → 检查日志（grounded_interpretation / relevance scoring elapsed / semantic prescore / family scoring probe）。
4. 判据：players 浮现专精玩家 + 主线带代表 + 并发 elapsed 达标。
