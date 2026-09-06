# 方案：US 外观（design）防侵权询检（B 方案自研版）

> 场景：卖家选品防侵权——上传产品图（或链接/品名）→ US 外观在先权利检索 → 视觉相似风险判断。
> 上游：`docs/superpowers/specs/2026-09-06-us-design-clearance-feasibility.md`（V1–V6 全部定案，链路已实证）。
> 立场：不建图像向量库/不依赖本地 GPU——召回靠 XHR 文本+`type=DESIGN`，判定靠多模态 LLM（复用 `analyze_patent_with_vision` 管道）。
> 范围：**首版 US-only**；CN/EM 见 §11 二期。

---

## 1. 背景与问题

### 1.1 用户与场景证据
- 卖家画像用户反复出现"产品防侵权"诉求（Air Toobz、遥控蛇、椭偏仪等样本），其中 7271… 已触及 **US 外观申请回执号**（30/076,484）——外观风险是卖家刚需；
- 现有管道只覆盖 utility（文本/著录检索、文本相似查重），**外观专利无可检索入口、无图对比能力**；
- 市场验证：智慧芽/睿观/麦德通等均以"上传产品图→外观相似→风险报告"为主形态（0.7 相似度阈值/TRO 标签为惯例）。

### 1.2 可行性实测结论（2026-09-06 两轮探针，`scripts/design_feasibility_probe.py`）
| 待验 | 结论 |
|---|---|
| V1 USPTO search | D 前缀 `patentNumber:"D504889"` 命中（纯 digits 404）；**无分类字段返回** → 仅 title 检索可用 |
| V2 Google XHR | **`type=DESIGN` 过滤生效**（toy snake 6286→270）；设计 id 带 S1 后缀（`USD504889S1`） |
| V3 附图 | 设计页主资源 = **PDF** 直链（200、203KB）→ 复用既有 PDF→页图→vision 管道 |
| V4 召回 | 纯词排序 top10 设计≈0 → **type=DESIGN 为硬前提**；EM/CN S 号在 XHR 可见 |
| V5 老件 | 15 年 term 推导自动失效过滤 → 风险场景仅涉 ≤15 年件 |
| V6 LOC | US 本土件**结构性无 LOC**（USPC-D 存储、API 不返分类）；LOC 定位 = L0 推断自检 + 未来 EM/CN 过滤 |

### 1.3 复用资产（零新基建清单）
- `analyze_patent_with_vision`（patent_analyzer.py:344，`[LONG_TASK] vision_provider/model` 可配）
- `text_extractor` PDF→页图→vision 扫描件管道（L2 直接复用）
- 上传/长任务：`_handle_file_upload_query`、进度/锚点/回执（需求#7）、失败单点出口（Phase A T5）、`failure_guidance`/预检门模式（T3/T4）、锚点追问引用（会话锚块）
- XHR 通道：PCT→WO spike 已验证的 Google XHR 检索模式

---

## 2. 目标、非目标与验收

### 2.1 目标
- **G1**：上传产品图 → 返回 US 外观在先权利 Top 候选（D 号/著录/有效期）
- **G2**：对候选做**视觉相似判定**（0–1 score + 三档风险 + 命中维度/依据句/差异点，可追溯）
- **G3**：产出风险报告（P1 文本报告；结果卡 risk 徽标列 P2），并写回执/锚点支持会话追问
- **G4**：全程 fail-open——单环失败降级为部分结果+原因，绝不整链崩溃；0 命中遵守"禁裸未找到"契约

### 2.2 非目标（YAGNI）
- CN/EM/全球外观库（XHR 可见但本期仅 US 件参与判定；EM/CN 记日志作二期信号）
- LOC/USPC 存储过滤（数据源结构性缺失，V6 裁定）；LOC 仅 L0 推断自检/报告标注
- 图像向量索引/自建以图搜图；图像反查桥（L1-b，二期）
- TRO 维权史、批量/全店铺扫描（睿观类能力，另行立项）
- 对比图并排前端（P2）；法律意见

### 2.3 验收
自动化（单测/集成）：
1. 15y/14y term 边界推导（2015-05-13 分界）与失效过滤/标注；
2. L1 阶梯生成（紧→松 ≤6 次）、`type=DESIGN` 必带、US 件过滤（EM 剔除样例）、503→退避→可重试提示、0 命中→报告契约文案；
3. L2 页解析抽 PDF URL（fixture HTML）、直链失败→"文本维度候选"降级；
4. L3 schema 校验（缺字段/score 越界/批拆分）与 risk≥0.7→high 映射；
5. 聚合纯函数：三档汇总、Top-5 高危、失效提示、digest/锚点结构（result_ids≤50）；
6. 长任务全链（mock L0–L3）：上传图→回执+锚点写；`long_task:fail` 单发；入口预检引导。
轨迹/部署 UAT：
- [ ] V4′：`type=DESIGN`+`country=US`（参数名实测）下 5 品名词 US 设计召回占比 → 定 L1 US 收窄方式与 Top-N 默认
- [ ] 真实产品图 1 件端到端人工核：召回件与判定档位是否可用、报告可读性
- [ ] Google 限流下的重试/降级表现（503 模拟或自然发生）
- [ ] 追问"第 N 件为什么高危/差异在哪"经锚点承接

---

## 3. 架构总览

```text
上传图像/文本(design 意图)
   │  入口判定 + 预检门(引导/拒绝 非任务)
   ▼
[L0 design_query]  视觉翻译 1×vision → {en_name, keywords, visual_features, locarno 自检, 澄清标志}
   │  (缺图/失败 → 文件名+文本降级; 多产品歧义 → 澄清轮)
   ▼
[L1 design_search] XHR q=阶梯 + type=DESIGN (+country=US UAT) → US 件过滤
   │  → 法律状态 15y 推导(失效过滤/标注) → Top-N 候选(默认 30)
   ▼
[L2 design_image]  每件: GP 页解析 patentimages PDF 直链 → PDF→页图(复用 text_extractor)
   │  (失败件降级"文本维度候选")
   ▼
[L3 design_judge]  产品图 × 2 件/批 vision → JudgeVerdict[] (score/risk/dims/basis/difference)
   ▼
[design_risk]      聚合纯函数: 三档汇总 → Top-5 高危 → 报告 markdown + digest/锚点
   ▼
SSE/结果卡/回执(需求#7 语义) → 会话追问可引用
```

---

## 4. 端到端数据流（一次询检）

```text
用户: 上传 product.jpg + "帮我查这个外观会不会侵权"
core: 图像+意图 → scenario=design_clearance → 任务入 long_task(created 消息)
celery execute_design_clearance:
  L0: vision(图) → {en_name:"Remote-control toy snake", kw:[remote controlled, toy snake,
      robotic snake], features:[分节蛇身, 遥控器], locarno:["21-01"], clarify:false}
  L1: q="remote controlled toy snake" type=DESIGN → ~14 件 → US 过滤 9 件
      → 状态推导: 2 件期满剔除(报告"相关但已失效") → 7 件候选(含 USD1,23x,xxxS1)
  L2: 逐件 PDF 直链抓取 + 页图(≤7 页/件) → 5 件成功, 2 件降级著录
  L3: 2 件/批 × 4 批 vision → verdicts
  risk 聚合: 1 件 high(score 0.82), 2 件 medium, 2 low
报告: 文本(高危逐件 D 号/档位/score/依据/差异 + 非法律意见声明)
      + 结果卡徽标 + digest(高风险 D 号≤5) → 锚点写 sess:{sid}:anchor
追问: "第 1 件和我的蛇头差别在哪" → 锚点块引用 → 展开该件依据句
```

---

## 5. 行为细节

### 5.1 入口判定与预检
- 触发条件：上传图像文件 且 意图词命中（规则词表：外观/侵权/查外观/设计专利/design…）；或纯文本含上述意图+品名描述 → 允许文本降级流
- 非图像且无可识别内容/意图 → `failure_guidance`（design 变体："请上传产品图或描述产品，我将检索美国外观专利"）
- 场景路由与预检门复用 T3/T6 模式（INSERT 前引导直答，不建任务）

### 5.2 L0 视觉翻译（design_query.py）
- Prompt 固定结构：输出 JSON `{en_name, keywords[], visual_features[], suggested_locarno[], needs_clarification}`；一次 vision 调用；产品图 ≤4 视图
- `needs_clarification` → SSE 澄清追问（等待用户补充）而非硬猜
- 失败 → 降级文本流（文件名字干+用户输入为 en_name 种子）

### 5.3 L1 XHR 检索（design_search.py）
- 查询：`q="{name} {kw1}"` + `type=DESIGN`（参数独立，已证）；`country=US` 待 V4′——可用则加参，不可用则结果侧 US 过滤
- 阶梯：`en_name+首特征词 → en_name → 单特征词(逐一) → 品类词`；≤6 次；0 命中换同义（L0 keywords 池）再放宽
- 礼貌退避：请求间隔 0.8–1.5s jitter；503/429 → 指数退避（1s/3s）重试 ≤2；仍败 → 失败走单点出口（"服务暂不可用，请稍后重试"）
- 结果过滤：`id 以 patent/USD 开头` 保留（EM/CN S 号记日志，不参与）；D 号/title/grant 日期/权利人保留
- **法律状态**：grant_date 起 15 年（2015-05-13 后申请）或 14 年有效；期满 → 过滤出候选集，汇总"相关但已失效 N 件"提示

### 5.4 L2 附图获取（design_image.py）
- 页 URL：`patents.google.com/patent/{id}/en` → 正则抽 `patentimages.storage.googleapis.com/…/USD….pdf`（首资源，V3 已证）
- PDF→页图：复用 `text_extractor` 既有 PDF→image 管道；页数 ≤7 截断；单件字节预算 ~1MB
- 降级：页 404/PDF 失败 → 候选保留为"文本维度候选（未视觉比对）"；批量失败不阻塞
- 并发 4 路；全程流式不落库

### 5.5 L3 视觉判定（design_judge.py）
- 批 = 产品图多视图 + 2 件候选图 → 1 次 vision → 批内逐件 JSON
- 判定口径固定句（prompt 常量）：ordinary observer 整体观感 + point of novelty；色彩非独立维度；依据必须指向图部位；每件给差异点
- 输出 schema 见 §6；score≥0.7 或 risk=high → 高危集
- 单批失败重试 1 次 → 仍败跳过并在报告注明"N 件未完成视觉比对"

### 5.6 聚合与报告（design_risk.py）
- 纯函数：输入 candidates + verdicts + term 推导 → {high[], medium[], low[], expired_hits[], report_md, digest_payload}
- 报告 markdown 结构（固定模板）：总览三档数 → 高危逐件（D 号/标题/档位/score/命中维度/依据句/差异点/链接）→ 相关但已失效 → 判定说明与免责声明
- digest：目标摘要 + 高危 D 号（≤5）；锚点 type=file target=上传名 result_ids=[D 号…]（≤50）

### 5.7 追问承接
- 完成锚点写会话 → 追问"第 N 件为什么高危"由锚块（需求#7 会话锚点）承接；`hydrate_session_task_messages` 补 created/completed 消息（含 digest）

---

## 6. 数据结构精确定义

```text
1) task params(MySQL input_params, scenario=design_clearance):
   { product_image_refs: [..], product_text: str, source: "us_design" }

2) 进程内:
   L0Product    { en_name, keywords[], visual_features[], suggested_locarno[],
                  needs_clarification }
   DesignCandidate { id, pub, title, grant_date, assignee,
                     status: "active"|"expired",  # 15y/14y 推导
                     pdf_url, pages_ok: bool }
   JudgeVerdict { d_number, score: 0..1, risk: "high"|"medium"|"low",
                  dims: [{name, score, note}], basis, difference }

3) 回执 digest(复用通道): 目标摘要 + 高危清单 ≤5 + "相关但已失效" N 件

4) 锚点(复用 sess:{sid}:anchor): target=上传文件名, type=file,
   result_ids=[高风险 D 号…]≤50

5) 前端: 结果卡 risk 徽标字段(P1 文本报告即可, 徽标 P2)
```

---

## 7. 错误处理（全部 fail-open，不破坏主链路）

1. 入口不明 → 引导直答（不建任务）
2. L0 失败/超时 → 文本降级；歧义 → 澄清轮
3. L1 503/429 → 退避重试 ≤2 → 失败单点出口（可重试提示）；0 命中 → 阶梯放宽 → 报告契约文案（禁裸"未找到"）
4. L2 单件失败 → "文本维度候选"标注；批量失败不阻塞
5. L3 批失败 → 重试 1 次 → 跳过并注明未比对件数
6. 聚合异常 → 内聚 try → 部分结果+原因，digest 仍写
7. 长任务未捕获异常 → worker 单点出口（notify+MySQL 成对、analytics 单发——Phase A T5 同款）
8. 会话锚点：成功才覆盖（需求#7 语义，失败不碰锚）

---

## 8. 代码改动清单（实施锚点）

| 文件 | 改动 | 对应 |
|---|---|---|
| `sources/design/__init__.py`（新） | 包入口 | — |
| `sources/design/design_query.py`（新） | L0 prompt/JSON 解析（纯函数可单测） | G1/§5.2 |
| `sources/design/design_search.py`（新） | XHR+type=DESIGN+US 过滤+退避 | G1/§5.3 |
| `sources/design/design_image.py`（新） | 页解析 PDF URL + 页图切分 | G1/§5.4 |
| `sources/design/design_judge.py`（新） | 分批 prompt+schema 校验 | G2/§5.5 |
| `sources/design/design_risk.py`（新） | term 推导/聚合/报告/digest（纯函数） | G3/§5.6 |
| `celery_worker.py` | `execute_design_clearance`（锚/回执/单点出口） | G3/§4 |
| `api_routes/core.py` | 图像+意图路由 + design 预检门 | §5.1 |
| `sources/long_task/status_manager.py` | （如需）design 引导变体 | §5.1 |
| 前端（P2） | 结果卡 risk 徽标/对比缩略 | G3 |

---

## 9. 测试计划

- `tests/test_design_risk.py`：15y/14y 边界（2015-05-13）、三档聚合、失效提示、digest 截断/锚结构
- `tests/test_design_search.py`：阶梯序列、type=DESIGN 必带、US 过滤（EM fixture 剔除）、503→退避→失败提示、0 命中契约文案（mock XHR）
- `tests/test_design_image.py`：fixture HTML → PDF URL 抽取；直链失败降级
- `tests/test_design_judge.py`：schema 校验（缺字段/score 越界）、2 件/批拆分、0.7 映射
- `tests/test_design_query.py`：JSON 容错、clarify 标志
- 长任务集成（mock L0–L3）：上传→回执+锚点；`long_task:fail` 单发；入口预检引导
- 回归红线：Phase A 套件零回归；pre-existing 失败画像不修；本机 `PYTHONUTF8=1`

---

## 10. 实施分期

- **P1（核心闭环）**：新域五模块 + executor + 路由/预检 → 图像与文本双入口可用；报告=文本 + 结果卡（徽标可后置）；每模块 TDD + 原子 commit
- **P2**：结果卡 risk 徽标与对比图并排（前端小改）；`country=US` 定案后 L1 收窄升级；EM/CN 路由；图像反查桥
- 收尾：全量回归 → 终审（code-review/opus）→ 服务器部署 → UAT（V4′ + 真实产品图端到端 + 限流表现）→ 需求列表回填

---

## 11. 风险与开放项

| 项 | 说明 | 处置 |
|---|---|---|
| L1 召回上限 | 纯文本+type=DESIGN，名称错位件可能漏（无向量索引固有代价） | 0 命中阶梯同义放宽；V4′ 定 US 收窄；二期图像反查桥 |
| Google 限流 | 503（第三轮实测） | 退避+间隔；失败可重试提示；备选 USPTO title 检索（V1 已证） |
| `country=US` 参数 | 未验（V4′） | UAT 首查；不可用则结果侧 US 过滤（已内置） |
| 老设计图覆盖 | >15 年件被 term 过滤 → 风险场景自动规避 | 不阻塞；日志留样本 |
| 视觉打分一致性 | 同图多评波动 | 报告只给三档+依据句，不给"最终法律结论"；置信标注 |
| vision 模型切换 | deepseek-v4-flash-vision-exp 候选 | `[LONG_TASK] vision_provider/model` 可配，L0/L3 不经硬编码 |
| 判定成本 | 30 件×2/批≈15 批 vision/询检 | 默认判定 Top-15 可配；报告注明范围 |
