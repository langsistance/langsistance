# 智慧问答批量检索结果分屏视图 — 技术设计文档

> 日期: 2026-08-13 | 分支: `feature/china-patent-analysis` | 状态: 待评审
> 参考形态: PatSnap Eureka（结果列表 + 详情面板）

---

## 一、背景与目标

### 1.1 现状与问题

智慧问答通过知识库知识条目 + 后端工具（USPTO 专利检索、USPTO 文档检索、Google Patents 检索等）完成检索。工具返回的结构化结果（`raw_items`）目前只做三件事：

1. 送给 LLM 格式化成 Markdown 总结，流式输出到聊天气泡（用户看到"一坨文字列表"）
2. 抽取 patent_id 存 Redis + 隐藏 SSE 事件（仅供后续长任务上下文）
3. 经 `result_export.py::build_result_artifacts()` 规范化为 CSV/XLSX artifact，前端只做下载

**结构化数据从未到达前端渲染层** —— 这是结果展示"不专业"的根因。

### 1.2 目标

参考 Eureka，将批量检索结果做成专业的分屏浏览体验：

- **独立结果页**（路由跳转）：三栏布局 —— 聊天窄侧边栏（可继续追问）+ 结果列表 + 详情面板
- 列表每行专利提供四个操作：**详情**（著录项+摘要）、**说明书**（全文）、**权利要求**、**审查历史分析**
- 审查历史分析拆两条 pipeline：🇺🇸 **美国专利审查历史**（prosecution）与 🌐 **跨国同族审查历史**（family）
- 聊天 ⇄ 结果页双向联动；会话恢复后结果可重建
- 复用现有 artifact 传输管道（CSV/XLSX 保持不变，下载能力不受影响）

### 1.3 v1 范围

- 数据源：USPTO 专利列表、Google Patents 结果、USPTO 文档列表
- 不含 CNIPA（zldsj）中国专利数据源（v2）

### 1.4 反目标

- 不在结果页做后端分页/重新检索（v1 排序筛选均为前端内存操作）
- 不新增 SSE 事件类型（列表数据走 artifact 管道）
- 不破坏现有行为：无 json artifact 的旧消息/旧会话优雅回退为纯下载按钮
- 说明书/权利要求不随 artifact 全量下发（按需接口拉取）

---

## 二、数据流全景

```
┌─ 后端（langsistance）───────────────────────────────────────────────┐
│                                                                   │
│ 用户提问 → 知识库路由 → 工具调用 → raw_items（现有流程，不动）          │
│                                                                   │
│ general_agent 现有 4 件事（全部保留）:                               │
│   ① raw_items → LLM 格式化总结 → SSE token 流（聊天里仍有摘要）      │
│   ② patent_ids → Redis（后续长任务上下文用）                        │
│   ③ patent_ids 隐藏 SSE 事件                                       │
│   ④ build_result_artifacts() → CSV/XLSX artifacts                │
│                                                                   │
│ 改动点①：④ 中新增第三种 json artifact  ←───────────── 唯一的列表数据源 │
│   {columns:[{key,label,role}], rows:[...], source}                │
│   → 走现有 artifact_start/chunk/end 通道（SSE 层零改动）             │
│                                                                   │
│ 改动点②：新增 3 个详情端点（Firebase 认证）                           │
│   GET  /patent/{source}/{patent_id}/spec    说明书分段              │
│   GET  /patent/{source}/{patent_id}/claims  权利要求               │
│   POST /long_task/submit                    审查历史：直接提交 Celery│
│        {scenario:"prosecution"|"family", patent_id, query, lang}   │
└───────────────────────────────────────────────────────────────────┘
                          │ SSE + REST
┌─ 前端（nextjs）────────────────────────────────────────────────────┐
│                                                                   │
│ 聊天页（改动点③）：收到 format=json 的 artifact → 解码重组 →          │
│   消息挂 results:{setId, source, columns, rows}                    │
│   气泡内渲染「结果卡片」：N 条结果摘要 + [在结果页查看] + 下载按钮      │
│                                                                   │
│ 结果页 /app/results?set={setId}&session_id={sid}（改动点④）:       │
│   ┌ 聊天窄栏 ┐ ┌ 结果列表 ┐ ┌ 详情面板 ┐                            │
│   │ 可继续追问 │ │ 1 标题    │ │ Tabs:   │                            │
│   │ 新检索结果 │ │ 2 标题    │ │ 详情    │ ← 行数据直接渲染，零接口   │
│   │ → 列表切换 │ │ ...      │ │ 说明书  │ ← 点击→调 spec 接口        │
│   └───────────┘ │ [下载]    │ │ 权利要求│ ← 点击→调 claims 接口      │
│                 └──────────┘ │ 审查历史│ ← 提交长任务→内嵌进度+报告  │
│                             └─────────┘                            │
│                                                                   │
│ 持久化（改动点⑤）：saveSessionMessages 增加 results 字段             │
│   （裁剪版：abstract 截 500 字、上限 50 行）→ 刷新/重开会话可恢复     │
└───────────────────────────────────────────────────────────────────┘
```

---

## 三、传输格式：json artifact + 列角色

### 3.1 为什么是 json artifact

复用现有 artifact 管道（`artifact_start`/`artifact_chunk`/`artifact_end` 原样透传 metadata 与 base64 分块），在 CSV/XLSX 之外新增 `format=json` 的第三种 artifact。前端解码后直接渲染，解码失败则回退现有下载按钮。

不选"前端解析 CSV/XLSX"的原因：

| 问题 | CSV 方案 | json 方案 |
|---|---|---|
| 哪列是"标题"？ | 按表头名猜（各数据源叫法不同，猜错即废） | `role` 显式标注，渲染层与数据源解耦 |
| 摘要含换行/引号/逗号 | CSV 解析边界 bug 多 | 原生 JSON，无歧义 |
| 嵌套字段（发明人列表、IPC） | 变成 JSON 字符串塞单元格 | 保留结构，行卡片直接渲染 |
| 详情按钮所需标识（申请号/公开号/来源） | 无 | 每行附带，驱动按钮可用态 |

### 3.2 artifact 契约

artifact metadata（走现有通道）：

```jsonc
{
  "artifact_id": "...-json",
  "format": "json",
  "filename": "CopiioAI_Result_20260813_120000.json",
  "mime_type": "application/json",
  "row_count": 20,
  "column_count": 12,
  "content": "<bytes>"   // 现有分块 base64 通道
}
```

content 解码后的 JSON 载荷：

```jsonc
{
  "source": "uspto",            // uspto | google_patents | uspto_documents
  "columns": [
    {"key": "patentTitle",  "label": "标题",   "role": "title"},
    {"key": "patentNumber", "label": "专利号", "role": "patent_id"},
    {"key": "applicationNumberText", "label": "申请号", "role": "application_number"},
    {"key": "assigneeEntityName",    "label": "申请人", "role": "assignee"},
    {"key": "filingDate",     "label": "申请日", "role": "filing_date"},
    {"key": "abstractText",   "label": "摘要",   "role": "abstract"},
    {"key": "未识别列",       "label": "未识别列", "role": "text"}
  ],
  "rows": [ /* 与 CSV/XLSX 完全同源的扁平行 */ ]
}
```

### 3.3 列角色推断 `infer_column_role(key)`

对已知字段路径做大小写不敏感的后缀/包含匹配。角色集合**封闭**：

| 角色 | 匹配字段（示例） | 用途 |
|---|---|---|
| `title` | patentTitle, inventionTitle, title | 行卡片标题 |
| `patent_id` | patentNumber, publicationNumber, pno | 元信息行 + 详情接口参数 |
| `application_number` | applicationNumberText, applicationNumber, apc | 美国审查历史按钮可用性（8 位数字） |
| `publication_number` | earliestPublicationNumber, pctPublicationNumber | 行卡片公开号 |
| `assignee` | assigneeEntityName, applicant, assignee | 元信息行 |
| `inventors` | inventors, inventorName | 详情 tab |
| `filing_date` | filingDate, applicationDate, 申请日 | 元信息行/排序 |
| `publication_date` | publicationDate, grantDate, 公开日 | 元信息行/排序 |
| `ipc` | ipcClass, cpcClass, ipc | 详情 tab |
| `abstract` | abstract, abstractText, 摘要 | 详情 tab + 行卡片摘要 |
| `document_title` | documentTitle, document_title | 文档列表行 |
| `document_date` | documentDate, document_date | 文档列表行 |
| `url` | document_url, downloadUrl, pdfUrl | PDF 原文/文档查看 |
| `text` | 其余全部 | 详情 tab 键值表 |

未识别的列 role=text，仍在"详情"tab 以键值表展示 —— 任何数据源都不丢字段。

### 3.4 生成时机与阈值

- `build_result_artifacts()` 增加 `source` 参数（调用方 general_agent 从工具 URL 判定）
- json artifact 与 CSV/XLSX 共用同一阈值（`RESULT_EXPORT_MIN_ROWS`，默认 6）：小结果集留在对话里，不开结果页

---

## 四、结果列表与详情面板交互

### 4.1 结果列表（中栏）

```
┌──────────────────────────────┐
│ 检索结果 20 条        [⬇Excel]│
│ 排序: 相关度 ▾  来源: 全部 ▾  │
├──────────────────────────────┤
│ ▸ 1 一种图像处理的方法及装置    │  ← 标题行（role:title）
│    US12000123B2 · 华为 · 2024 │  ← 元信息行（patent_id·assignee·日期）
│    [详情][说明书][权要][审查历史]│  ← 操作按钮
│ ▸ 2 深度学习目标检测网络...     │
│    CN118...A · 腾讯 · 2023    │
│    [详情][说明书][权要][审查历史]│
└──────────────────────────────┘
```

- **行卡片**：标题（点击=详情）+ 元信息行（字段由 role 驱动，缺省省略）+ 四个按钮
- **按钮态**：`审查历史` 无 8 位美国申请号时拆分为仅"🌐 同族"可用、"🇺🇸 美国"置灰 + tooltip；点击任意按钮 = 切换详情面板对应 tab + 懒加载
- **高亮当前行**：详情面板与选中行联动
- **排序/筛选**：v1 前端内存（相关度=原始顺序、日期、申请人）+ 来源筛选；后端重新检索是 v2
- **加载/失败态**：说明书/权要首次点击骨架屏，失败显示重试按钮

### 4.2 详情面板（右栏）

Tabs：`详情` | `说明书` | `权利要求` | `审查历史`

| Tab | 内容 | 数据来源 |
|---|---|---|
| 详情 | 著录项卡片（专利号/申请号/申请人/发明人/申请日/公开日/IPC）+ 摘要全文 | 行数据直接渲染，零接口 |
| 说明书 | 分段全文（技术领域/背景技术/发明内容/具体实施方式等），段间目录跳转，顶部 [PDF 原文] 链接（有 url 列时） | 进入 tab 时调 `GET /patent/{source}/{patent_id}/spec`，面板内缓存 |
| 权利要求 | 权利要求列表，独立权要高亮标记，一键复制 | 进入 tab 时调 `GET /patent/{source}/{patent_id}/claims` |
| 审查历史 | 两个入口卡片 + 提交后内嵌 LongTaskProgress + 完成后 Markdown 报告 + [下载 PDF/Word] | `POST /long_task/submit` + 现有 status/report 轮询 |

**审查历史 tab 的两个入口：**

| 入口 | scenario | 可用条件 |
|---|---|---|
| 🇺🇸 美国审查历史分析 | prosecution | 行数据含 8 位美国申请号 |
| 🌐 跨国同族审查历史分析 | family | 任意专利（EPO 自动解析同族） |

- USPTO 专利行 → 两个都亮；Google Patents 结果（CN/EP/JP 行）→ 仅同族可用
- 每次点击生成独立 task；同一专利的两个分析互不覆盖，可切换查看
- 提交后 task_id 挂入消息 `long_task_ids`（复用现有字段），恢复会话时现有轮询自动接管

### 4.3 USPTO 文档列表形态

文档检索工具的结果行 role 为 `document_title`/`document_date`/`url`：

- 列表行卡片：文档标题 + 日期，按钮 `[详情]` `[查看原文]`（复用 `/uspto/download` 代理或新窗口打开 url）
- 详情面板：单 tab "文档信息"（标题/日期/来源元数据 + 原文下载按钮）
- 同一列表组件靠 column role 区分 —— 专利行与文档行可混在同一结果集

### 4.4 聊天侧边栏联动（双向）

- 窄栏复用现有聊天输入 + send 逻辑，流式过程不阻塞结果页
- 新消息产生新 json artifact → 结果列表切换到新结果集，聊天同步出现新结果卡片
- 点击聊天中的旧结果卡片 → 结果页切回对应结果集（`?set=` 同步）
- 直接访问 `/app/results`（无 session）→ 空态引导"回对话页发起检索"

---

## 五、API 契约

### 5.1 GET /patent/{source}/{patent_id}/spec

```
source ∈ uspto | google_patents
patent_id: 公开号或申请号（如 US12000123B2、17429113）

200: {success: true, sections: [{heading, paragraphs[]}], source_url}
502: {error}    // 上游失败，日志留详情
```

实现：uspto → `uspto_download.py` + `text_extractor.py`；google_patents → `google_patents_client.query_description()`。无自然分段时按 `[0001]` 编号段切分。

### 5.2 GET /patent/{source}/{patent_id}/claims

```
200: {success: true, claims: [{number, text, independent}]}
501: {error: "权利要求暂不可用，请通过 PDF 原文查看"}
```

google_patents → `query_claims()`；uspto → 全文下载后按 "Claims"/"What is claimed" 解析（尽力而为，失败 501 诚实降级）。

### 5.3 POST /long_task/submit

```jsonc
{ "scenario": "prosecution" | "family", "patent_id": "...", "query": "...", "lang": "zh" | "en" }
→ {success: true, task_id, status: "queued"}
400: scenario/patent_id 非法（如 prosecution 但无 8 位美国申请号）
429: 队列满（附 queue_info）
401: 未认证
```

复用现有 dispatch 逻辑（`_dispatch_from_mysql` 同层）；后续 patent_v1 MCP 路由（API key 认证）包同一服务层，不重复实现。提交后前端轮询现有 `GET /long_task/{task_id}/status`、下载走 `GET /long_task/{task_id}/report`。

---

## 六、错误处理矩阵

| 场景 | 后端 | 前端 |
|---|---|---|
| spec/claims 拉取失败 | 502+error（日志留详情） | tab 内错误卡 + [重试]；有 source_url 时给 [PDF 原文] 兜底 |
| 美国审查按钮不可用 | 400 校验 | 按钮置灰 + tooltip（前后端双保险） |
| 长任务队列满 | 429 + queue_info | tab 内排队提示 |
| json artifact 缺失/解析失败 | — | 回退纯下载按钮（现有行为，零破坏） |
| 旧会话无 results 字段 | — | 不渲染卡片，仅下载按钮 |
| 说明书无自然分段 | 按 [0001] 编号段切分 | 渲染段落列表 |
| 权要解析失败 | 501 | tab 内提示 + [PDF 原文] |

---

## 七、持久化

- `saveSessionMessages` 增 `results` 字段：`{setId, source, columns(含role), rows}`，裁剪规则：abstract 截 500 字、上限 50 行（预估 ≤ 100KB，MySQL JSON 可承受）
- 结果页刷新恢复：`/app/results?set={setId}&session_id={sid}` → `getSession` → 按 setId 找回消息 → 水合
- 审查历史 task_id 挂入消息 `long_task_ids`（复用现有字段与恢复轮询）
- setId = json artifact 的 artifact_id：一次对话多次检索 → 多张结果卡片，互不覆盖

---

## 八、改动文件清单

### 后端（langsistance）

| 文件 | 变更 |
|---|---|
| `sources/result_export.py` | 新增 `infer_column_role()`、json artifact 构建；`build_result_artifacts()` 增 `source` 参数并返回 json 项 |
| `sources/agents/general_agent.py` | 调用点传 `source`（从工具 URL 判定 uspto/google_patents/documents） |
| `api_routes/patent_detail.py`（新） | spec / claims 两个端点，包装现有客户端 |
| `api_routes/long_task.py` | 新增 `POST /long_task/submit`（复用 dispatch 层） |
| `api.py` | 注册 patent_detail 路由 |
| `tests/` | 角色推断、json artifact、spec/claims 端点（mock 客户端）、submit 端点单测 |

### 前端（nextjs）

| 文件 | 变更 |
|---|---|
| `lib/chatSession.js` | json artifact 解码重组 → `message.results` |
| `lib/results.js`（新） | 行模型构建（role→卡片渲染）、持久化裁剪 prune、结果集切换 |
| `services/api.ts` | fetchSpec / fetchClaims / submitLongTask |
| `components/app/ResultCard.tsx`（新） | 聊天气泡内结果卡片 |
| `app/app/(auth)/results/page.tsx`（新） | 结果页三栏布局 |
| `components/app/results/`（新） | ResultList / ResultRow / DetailPanel / SpecTab / ClaimsTab / ProsecutionTab / DocTab |
| `app/app/(auth)/chat/page.tsx` | 结果卡片渲染 + 跳转入口 |
| `contexts/ChatContext.tsx` | results 状态在 chat 页与 results 页间共享 |

---

## 九、测试策略

- **后端 pytest**：`infer_column_role` 全角色覆盖；json artifact 构建（含阈值、source 传递）；spec/claims 端点（mock uspto/google_patents 客户端，含 502/501 路径）；submit 端点（400/401/429 + 正常提交）
- **前端单测**：json artifact 解析（含损坏数据回退）、results 裁剪、行卡片渲染（role 驱动、缺失字段省略）、按钮可用态（美国审查 8 位号判定）
- **端到端手动验证**：检索 → 结果卡片 → 结果页 → 详情/说明书/权要 → 审查历史提交 → 进度 → 报告；刷新页面恢复；旧会话回退

---

## 十、v1 范围外

- ❌ CNIPA（zldsj）中国专利数据源接入结果页
- ❌ 结果页后端分页/重新检索（排序筛选 v1 仅前端内存）
- ❌ AI 摘要、相似专利、引证关系（Eureka 高级功能，依赖额外数据源）
- ❌ 说明书/权要全文 AI 翻译（先展示原文）
- ❌ 结果集跨用户分享

---

## 十一、设计决策记录

| # | 决策 | 原因 |
|---|---|---|
| 1 | 独立结果页（路由跳转）+ 聊天窄侧边栏 | 用户选定；最接近 Eureka，且不打断对话 |
| 2 | 列表数据走 json artifact（复用现有管道） | 复用 chunk 重组逻辑；SSE 层零改动；旧消息优雅回退 |
| 3 | 列角色（role）标注而非前端解析 CSV | 渲染层与数据源解耦；无解析歧义 |
| 4 | 详情 tab 用行数据零接口 | 著录项已在检索结果中，避免多余接口 |
| 5 | 说明书/权要按需拉取 | 全量下发会撑爆 SSE；符合"点击按钮调接口"交互 |
| 6 | 审查历史拆 prosecution/family 两条 pipeline | 用户指定；两者数据源与耗时不同，需独立入口 |
| 7 | submit 端点放 long_task 路由（Firebase 认证） | 复用现有 dispatch 与轮询；patent_v1 MCP 未来包同一服务层 |
| 8 | results 裁剪持久化到会话消息 | 复用现有 session 存储与恢复链路 |
| 9 | v1 排序筛选仅前端内存 | 后端重新检索需驱动 LLM 再调工具，链路复杂，v2 再做 |
