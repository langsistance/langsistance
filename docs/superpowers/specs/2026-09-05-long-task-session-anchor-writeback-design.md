# 需求 #7 方案：长任务结果回填会话 + 双入口归一 + 会话锚点（C 范围 · 服务端权威路线）

- 日期：2026-09-05
- 状态：设计待审（未实现）
- 对应需求：用户增长需求列表 #7（chat 与 long_task 双入口归一、长任务完成回填会话），联动 #15（长任务结果摘要）、#25（检索式结构化交付/会话锚点重建）
- 证据用户：9565…（2026-09-04 上传→追问断链）、7271…/7577…/1835…（双入口两套结果不互通）

---

## 1. 背景与问题

### 1.1 问题一：上传→追问断链（9565… 09-04 现场）

用户上传文件 B（一种双吸附头旋转纠偏机构）走 file_upload 长任务，任务 62s 完成。随后在同一会话追问「给出我相似文件的申请号 或者名字」与「给到我检索式」。现场表现：

- agent 上下文里**没有文件 B 的内容、也没有任务结果**（search_interpretation 输出"未提供待比对方案"，然后回退上一个主题（围挡/PCB 吸附装置）的关键词重搜）；
- 检索式输出沿用旧主题静态词库（零工具调用）。

根因：任务完成时只写回「报告文件」，**相似检索结果 TopN 与目标文件文本摘要从未进入任何会话载体**；会话侧也没有"当前待比对对象"的持久化指针。

### 1.2 问题二：双入口不互通（7271…/7577…/1835…）

- chat 检索结果落 `lt:conv:{user}:patent_ids`（Redis，TTL 1h）；
- 上传查重/分析任务结果落任务状态（Redis `lt:*`）与报告文件；
- 两层互不相通：同一诉求先对话查、再上传查 = 跑两遍、两套结果（1835… 并行"50 条 vs 1 条"即为此例）。

### 1.3 现状中已具备、本方案复用的资产（M1，2026-09-03）

| 资产 | 位置 | 说明 |
|---|---|---|
| 任务生命周期消息写回会话 | `sources/long_task/task_messages.py` `append_task_message` | 把 created/completed/failed 写为带 `meta{kind:long_task,task_id,seq}` 的 assistant 消息，固化进 `conversations.messages` |
| 完成 digest | `task_messages.py` `build_result_digest`（DIGEST_MAX_CHARS=9000） | 从 Redis sticky `result_summary` 截取头部 |
| 追问轮 hydrate | `api_routes/core.py:1131-1140` | 有 `request.session_id` 时把前端未见过的任务消息补进 agent_history（去重键 task_id#seq） |
| 完成写回统一入口 | `sources/long_task/status_manager.py:94-125` `set_task_completed` | 置 completed + 追写 digest 消息，失败静默 |
| 会话存储 | MySQL `conversations`（messages JSON、long_task_ids），前端每轮全量背回 | `api_routes/session.py:42-206` |
| conversation_refs 读历史消息号 | `api_routes/core.py:499-532` | 读 assistant 消息顶层 `patent_ids`/`patent_data`（含 spec_text>100 转 patent_texts）|
| chat 检索号记忆 | `sources/agents/general_agent.py:77,190-203,265-269` | `lt:conv:{user}:patent_ids` 写入与注入"前序检索命中专利号"块 |
| LLM 场景分类器 + 低置信 chat_fallback | `api_routes/core.py:191-359,1215-1249` | 7 场景；regex 兜底门（2026-09-03 事故驱动）|

### 1.4 缺口清单（本方案要补的洞）

1. **完成上报不统一**：部分 executor 完成时只传 `report_files`（如 family 路径 `celery_worker.py:2862`），不带 `patent_ids`/结果摘要 → digest 是空壳（需求 #15 未闭合）；
2. **file_upload 相似检索结果无结构化载体**：TopN（申请号/标题/来源/关联度）不写回任何会话可见/可引用的地方；
3. **无"当前对象"指针**：解释器/检索式生成器只能从散落历史猜，09-04 猜错；
4. **双入口无会话级合流**：chat ids 与任务结果互相不可见，同诉求重复起任务。

## 2. 目标、非目标与验收

### 2.1 目标

1. 任务完成后，同一会话追问（申请号/名字/继续查/检索式）直接引用该文件的结果与主题，不回退旧主题重搜；
2. chat 检索与上传查重结果按会话合流，同诉求不重复起任务；
3. 会话"当前待比对对象"成为一等状态（锚点），供解释器与检索式重建程序化读取（#25 数据地基）。

### 2.2 非目标（YAGNI）

- 不做前端新消息类型/新 UI（会话消息渲染既有能力足够；可选锚点指示条列为后续）；
- 不做跨会话锚点记忆（lt:conv 已承担跨会话裸号记忆；锚点仅会话级）；
- 不做检索式重建的完整产品化交互（#25 本体单独排期，本方案交付最小可用"按锚点生成阶梯"）；
- 不做公司/申请人聚合分析等 #10 内容。

### 2.3 验收（09-04 轨迹复跑清单）

- [ ] 传文件 B → 任务完成 → 追问「给出我相似文件的申请号」→ 答复直接列任务结果 TopN（含申请号/标题），无"未提供待比对方案"误判、无旧主题词重搜；
- [ ] 追问「给到我检索式」→ 检索式按文件 B 主题生成（ti/ab/clm + 载体词阶梯）；
- [ ] 同一会话先 chat 查过某号再上传同文件 → 出现复用提示而非静默二次起任务；
- [ ] 上传任务完成事件可追溯（日志可验证 digest 消息写回 + 锚点键存在）。

## 3. 架构总览

```
档案层 MySQL（永久）
  conversations.messages   ← 任务回执消息（人类可读 digest + patent_data）
  long_tasks               ← 任务行（已有）
当前态层 Redis（可丢可重建）
  已有  lt:{task_id}:status        （24h）
  已有  lt:conv:{user}:patent_ids  （1h）
  新增  sess:{session_id}:anchor   （24h 滑动，≤2KB）
注入/消费层（后端代码）
  hydrate（core.py:1131）补回执 → system prompt 锚点块（general_agent）
  → 解释器/分类器/检索式生成器读取（同一份 system prompt，零新接口）
```

分层原则：**Redis = 会话级快速态（当前指针：可丢、可重建、last-write-wins）**；**MySQL = 永久档案（回执消息固化结果，真正的资产）**。锚点键过期可从档案中的最近回执**重建**——降级链见 §5.3。

## 4. 端到端数据流

```
① 上传/长任务提交（core.py 现有：会话复用或新建，921-957 / 1199-1289）
② celery 各 executor 完成 → 统一 set_task_completed(task_id, report_files,
     patent_ids, rows_digest, anchor_payload)
③ status_manager.set_task_completed（增强）
   ├─ Redis lt:{task_id}:status → completed + patent_ids + result_summary（已有逻辑）
   ├─ append_task_message('completed', 增强 digest)          → MySQL messages（档案）
   └─ session_anchor.write(session_id, anchor_payload)       → Redis 锚点键（当前态）
④ 追问轮 /query_stream（core.py generate）
   ├─ hydrate_session_task_messages（已有 1131）→ 补入未见回执
   ├─ session_anchor.load(session_id) → 注入 system prompt 锚点块
   └─ 解释器/分类器/检索式生成器读锚点块（对象查找顺序见 §5.1）
⑤ 双入口归一（会话级视图 = 锚点 result_ids ∪ 回执 patent_ids ∪ lt:conv）
   R1/R2/R3 规则见 §5.2
```

## 5. 行为细节

### 5.1 解释器对象查找顺序（写死判定逻辑）

本轮显式对象（新号/新文件/新上传） → 锚点块 → previous conversation → 反问用户。
"未提供待比对方案"仅在**前三者全空**时输出（消除 09-04 误判）。

追问轮行为矩阵：

| 本轮提问（隐含指代） | 判定路径 | 改造后行为 |
|---|---|---|
| 「给出相似文件的申请号/名字」 | 锚点 type=file 命中 | 直接引用锚点 result_ids + 回执 patent_data 列清单，不重搜 |
| 「这个呢」「还有吗」「按这个查 US」 | 锚点命中 | 以 target_summary 重建阶梯扩展检索 |
| 「给到我检索式」 | 检索式意图 | 按锚点主题生成 ti/ab/clm + 载体词阶梯（#25 最小联动）|
| 点名新专利号/新上传 | 显式新对象 | 走 direct_ids/新任务，完成后覆盖锚点 |
| 全新主题提问 | 无锚点关联 | 锚点仅参考不强制（现状逻辑不变）|

### 5.2 双入口归一（三个规则）

- **R1 去重起任务**：同会话内再传规范化后同名文件且距上次任务完成 <10 分钟 → 不入队，直接回引上次回执结果；
- **R2 chat 查过再上传**：上传 query 携带的专利号在本会话回执/lt:conv 已有 → 任务照跑，回执消息附加"本会话已查过 X（命中 N 件）"提示；
- **R3 追问引用顺序**：conversation_refs 取号顺序 = 回执消息 patent_ids（含锚点）> lt:conv，杜绝跨主题扫历史号（2026-09-03 事故模式）。

### 5.3 锚点生命周期与降级链

```
写入：任务完成 → session_anchor.write（last-write-wins，并发取新）
续期：chat 追问轮 load 成功 → 滑动续 24h
失效：自然过期 / 新任务完成覆盖

读取链（session_anchor.load 内部）：
  ① 锚点键命中 → 返回
  ② 键缺失/过期 → rebuild_from_session_messages(session_id)
     从 conversations.messages 找最近一条 meta.kind=long_task 的
     completed 回执 → 用其 content/patent_ids/patent_data 重建
  ③ 连回执都没有 → 返回 None → 行为 = 现状（解释器退回历史/反问）
     —— 永不比现状差
```

## 6. 数据结构精确定义

### 6.1 新增 Redis 锚点键（唯一新增存储）

```
键:  sess:{session_id}:anchor
TTL: 24h 滑动（每次 chat 追问轮成功 load 时续期）
值:  JSON，总大小硬上限 ≈2KB
```

```json
{
  "version": 1,
  "type": "file" | "number" | "topic",
  "session_id": "sess_xxx",
  "task_id": "lt_xxx",
  "target": "一种双吸附头旋转纠偏机构_机械_实用新型",   // ≤200 字
  "target_summary": "…抽取文本/权要首段 ≤600 字…",      // 语义锚
  "source": "cnipa",
  "result_ids": ["CN217415278U", "CN116008615A"],       // ≤50 条
  "result_titles": { "CN217415278U": "…" },             // ≤50 × ≤80 字，可选
  "updated_at": 1788…
}
```

写入点：任务完成侧；`session_id` 经 `task_messages._lookup_task_session(task_id)` 解析（已有）。`target_summary` 来自执行器手头已有的抽取文本（file_upload 路径 `text_extractor` 产物），**不二次抽取**。

### 6.2 digest 回执消息增强（改 task_messages.py）

完成回执 content 结构（人类可读，同时是降级重建源）：

```
任务已完成 —— 目标：<文件名/专利号>（来源 cnipa）
共 N 件相似/相关结果：
| # | 申请号 | 标题 | 来源 | 关联度 |
…（≤9000 字截断，沿用 _truncate_markdown）
（完整结果请在结果面板查看/下载）
```

消息顶层字段（保持既有契约 + 结构化扩展）：
- `patent_ids: [...]`（已有）
- 新增 `patent_data: [{patent_id, title, source, score}]`（≤50 条）——与 core.py:523-531 已读形状一致，但**不带 spec_text**（不触发专利文本分析误入，仅作引用清单）

### 6.3 system prompt 锚点块（改 general_agent.py 提示词构建器）

与"前序检索命中专利号"块同一注入区（general_agent.py:131-203），独立小节、显式声明参考属性：

```
## 当前会话任务锚点（本会话最近完成的上传/分析目标；仅当用户指代它时使用）
- 目标：<target>（<type>，来源 <source>，完成 <时间>）
- 结果：<result_ids 前 20 个>（共 N 件）
- 摘要：<target_summary 前 200 字>
```

约束（与既有 guidance 语言中性规则一致）：块文案不得固化领域词/示例提问词（`reject-query-specific-synonym-hardcoding` 记忆规则同样适用于锚点块模板）。

## 7. 错误处理（全部静默降级，不破坏主链路）

| 故障 | 行为 |
|---|---|
| 锚点 Redis 写失败 | 仅日志；回执已固化 MySQL → 降级链②可重建 |
| MySQL 读失败（rebuild 用）| 返回 None → 现状行为 |
| 载荷超限 | anchor ≤2KB 硬约束（生成侧截断）；digest 沿用 9000 截断；patent_data ≤50 条 |
| 并发任务完成 | 锚点 last-write-wins 取新 |
| 旧 executor 未带新载荷 | 参数缺省 None → 不回写锚点，行为不变 |

## 8. 代码改动清单（实施锚点）

| 文件 | 改动 |
|---|---|
| `sources/long_task/session_anchor.py`（新） | write / load（滑动续期）/ rebuild_from_session_messages / build_block 四函数 |
| `sources/long_task/task_messages.py` | build_result_digest 增强（目标行 + TopN 表格）；append 支持 patent_data |
| `sources/long_task/status_manager.py` | set_task_completed 接受结构化载荷并调用 session_anchor.write（session 经 lookup 解析）|
| `celery_worker.py` | 各 executor 完成点补齐 patent_ids/rows_digest/anchor_payload（重点 2862 类缺口；1539 已有雏形）|
| `sources/agents/general_agent.py` | 锚点块注入（131-203 注入区）；检索式/解释器消费侧读取 |
| `api_routes/core.py` | 解释器对象查找顺序（§5.1）；conversation_refs 取号顺序 R3；上传提交前 R1/R2 检测；hydrate 调用点补 load_session_anchor 续期 |
| `tests/` | 见 §9 |

## 9. 测试计划

**新增单测**
- `test_session_anchor.py`：write / load 滑动续期 / rebuild-from-receipt / build_block 截断与空值边界
- R1/R2 复用检测规则单测（mock DB/Redis）
- conversation_refs 引用回执 patent_ids 不回搜（mock 分类管线）

**扩展**
- `test_task_messages.py`：增强 digest 含目标行 + TopN 表格；patent_data ≤50 截断
- `test_context_injection.py`：锚点块注入（有/无锚点；全新主题不污染；语言中性约束）

**回归**：test_task_messages / test_context_injection / test_number_resolve / test_seller_real_inputs 保持全绿。

**集成 UAT**：09-04 轨迹复跑清单（§2.3），本地 mock celery 或服务器分阶段验证。

## 10. 实施分期（每期独立可合入）

| 期 | 内容 | 交付物 |
|---|---|---|
| P1 | 完成上报统一 + digest 增强 | celery 各 executor 完成带 patent_ids/摘要；status_manager/task_messages 增强 |
| P2 | 锚点键 + 注入 | session_anchor.py 四函数、完成路径写锚点、general_agent 锚点块 |
| P3 | 追问引用闭环 | 解释器查找顺序、conversation_refs R3；09-04 场景 UAT |
| P4 | 双入口复用检测 R1/R2 | 去重起任务 + 提示 |
| P5 | 检索式锚点联动（#25 最小版） | 检索式生成器读锚点生成概念阶梯 |

P1+P2 完成后数据层闭环就绪（回执与锚点可见、可被注入）；P3 完成引用顺序固化并通过 09-04 场景 UAT（此时 #7 核心 + #15 达成）；P4/P5 为 C 范围余项。

## 11. 风险与开放项

- **部署版本差**：M1（task_messages）为 09-03 代码，线上服务器版本未确认——P1 开工前先做 M0 审计（现场日志确认 digest 消息/锚点缺失的实际原因）；
- **digest 消息膨胀**：回执消息内容增长 → hydrate 注入量与 conversation_history 每轮背回量上升；沿用 9000 截断并在 §6.2 保证表格压缩；
- **检索式最小联动边界**：P5 的"概念阶梯生成"需遵守通用性规则（禁止固化单次提问词汇），生成器只读锚点 target_summary 的结构化概念，不做硬编码词表；
- **多会话/多任务歧义**：锚点 last-write-wins + 降级链②已覆盖；不做对象切换 UI（非目标）。
