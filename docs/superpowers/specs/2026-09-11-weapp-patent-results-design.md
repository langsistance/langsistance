# 小程序专利结果列表（对齐 web 的结果面板）

- 日期：2026-09-11（brainstorm 定稿）
- 状态：设计已确认，待实现
- 分支：`feat/weapp`
- 前序：M3 收尾包（`2026-09-11-weapp-m3-design.md`，已交付）
- 被本设计取代的非目标：`2026-09-10-weapp-deepseek-ux-design.md` §2 把「完整结果面板」列为非目标；本轮开始做

---

## 1. 背景与问题

小程序里助手回答的下方只有一排**专利号 chip，点击只做复制**——看不到任何详情。

**这些号码是「猜」出来的，不是真数据。** `frontend/weapp/src/services/chat.ts:56-65` 用正则从**回答正文**里抠：

```js
const PATENT_RE = /(CN\d{7,12}[A-Z]?|\b\d{8}\b)/g   // 最多 20 个
```

所以只有号码、没有标题申请人——因为它压根没读过真实数据。`chat.ts:54-55` 的注释自己写着这是过渡方案。

**而真实数据其实早就到了。** 后端在回答末尾发一个 **json 工件**，载荷是完整结构化结果：

```json
{ "source": "uspto",
  "columns": [{"key": "applicationMetaData.patentTitle", "label": "标题", "role": "title"}, ...],
  "rows": [ { "applicationMetaData.patentTitle": "…", "...": "…" } ] }
```

小程序**已经收到它**——`pages/chat/index.tsx:410-423` 的 `onArtifactsReady` 把包括 json 在内的全部工件都挂进了 `m.artifacts`。只是 `downloadableArtifacts`（`index.tsx:80-82`）在**渲染时**把它过滤掉了。数据一直在内存里躺着，没人消费。

本设计就是给这份数据做出消费方。

---

## 2. 目标与非目标

**目标**

1. 回答下方给出**真实**的专利结果入口（有 json 工件时才出现）
2. 独立结果页：列表展示**标题 + meta**（专利号 · 申请号 · 公开号 · 申请人 · 公开日）
3. 详情：全部非空字段的键值表
4. 说明书：拉 `spec` 接口的 PDF 并打开
5. 权利要求：结构化优先渲染，无结构化回退 PDF
6. 本地持久化，重开小程序后历史会话仍能回看结果

**非目标**

- **审查历史（prosecution）**——web 的那个按钮不发详情接口，而是发起一个**新的长任务**（`frontend/nextjs/app/app/(auth)/app/…/page.tsx:173-191`）。那是「开始新分析」，不是「看详情」，与本轮目标不同
- **排序 / 筛选 / 搜索 / 分页 / 虚拟滚动**——web 这五样一个都没有（`ResultList.tsx:41-46` 纯 `rows.map`，无 sort；无筛选控件；无搜索框；无分页；普通滚动容器）。对齐它，不发明
- **分享结果列表给外部人**——与 M3 已确立的权限模型冲突（会话有归属校验）
- 不修改后端任何接口

---

## 3. 调研结论（设计依据）

以下均为代码核实事实，是本设计的立足点。

### 3.1 json 工件载荷

`sources/result_export.py:654-665`：

```python
json_payload = {"source": source,
                "columns": [{"key": col, "label": uspto_field_label(col, lang),
                             "role": infer_column_role(col)} for col in columns],
                "rows": rows}
```

- 顶层**只有 3 个键**。行数是**行内**的 `row_count`/`column_count`（在 SSE 元信息里，不在 payload 内）
- `rows` 一行是 `dict[str, str]`，值一律已 `_stringify_cell` 成字符串（`result_export.py:133-138`）；None → `""`
- **无行数上限**，只有下限 `DEFAULT_EXPORT_MIN_ROWS = 1`（`result_export.py:17`）
- 分片粒度 `ARTIFACT_CHUNK_BYTES = 262144`（256 KB），base64 后逐帧发（`sse_callback.py:11`、`:60-67`）

### 3.2 role 的取值与用途

`infer_column_role`（`result_export.py:90-100`）按**最后一个路径段**、大小写不敏感、**首个匹配胜出**，从 `_ROLE_SUFFIXES`（`:45-87`）推断，共 14 个 role。

本设计用到其中 8 个：

| role | meta 行 / 详情用途 |
|---|---|
| `title` | 列表项标题 |
| `patent_id` | meta |
| `application_number` | meta |
| `publication_number` | meta |
| `assignee` | meta |
| `publication_date` | meta |
| `url` | 说明书 tab 的 PDF 直链（`_lift_download_url` 优先提升 PDF，`:103-130`） |
| `abstract` | 详情（需截断） |

其余 role（`document_title` / `document_date` / `inventors` / `filing_date` / `ipc` / `text`）在详情里**原样全列**，不进 meta。

**meta 行的构成与顺序对齐 web**：`frontend/nextjs/lib/results.js:34` 的 `META_ROLES`，且**只收录有值的**（`:48-54`）。

### 3.3 web 端消费方式

| 环节 | 位置 |
|---|---|
| 逐 chunk 解码 base64 → 文本 | `frontend/nextjs/lib/chatSession.js:134-161` |
| 解成 `{setId, source, columns, rows}` | `chatSession.js:177-195` |
| 列表渲染 | `components/app/results/ResultList.tsx` |
| 单行渲染（标题 + meta + 动作） | `components/app/results/ResultRow.tsx:31-65` |
| 详情面板（全字段键值表） | `components/app/results/DetailPanel.tsx:53-71` |
| 持久化 | `lib/resultsStore.js`，键 `copiioai_results`，结构 `{sets, index}` |

**关键坑（必须照搬）**：base64 解码要**逐 chunk 独立解再合并字节**，不能先拼字符串再解——256KB 的分片边界会把 base64 三元组切断。web 的注释还在（`chatSession.js:134-161`）。

### 3.4 详情接口

`api_routes/patent_detail.py`：

| 端点 | 返回 |
|---|---|
| `GET /patent/{source}/{patent_id}/spec[?pub_date=]` | `{success, pdf_url}` |
| `GET /patent/{source}/{patent_id}/claims` | `{success, claims?: [{number,text,status,independent}], pdf_url?}` |

约束：
- 需要 Firebase token
- `source ∈ {uspto, google_patents, baiten}`（`:30`）
- `patent_id` 非空且 ≤ 40 字符
- **失败返回 HTTP 200 + `{success:false, message}`**，不是 5xx——注释说明 Cloudflare 会替换源站 5xx 页面并导致 CORS 失败（`:774-778`）。**调用方必须按 `success` 判定，不能只看状态码**

**参数选择（对齐 web）**：spec 用 `patentId || applicationNumber`（`SpecTab.tsx:19`）；claims 用 `applicationNumber || patentId`（`ClaimsTab.tsx:21`）；两者都传**行级** `source`（优先于 payload 级，`results.js:66-70`）。

### 3.5 小程序现状的缺口

| 缺口 | 证据 |
|---|---|
| json 工件在渲染层被过滤 | `pages/chat/index.tsx:80-82` |
| 数据**从不持久化** | `saveMessages` 只传 `{role, content}`（`index.tsx:523-527`）；`selectSession` 只重建 `{role, content}`（`index.tsx:309-315`） |
| 内存**从不释放** | json 的完整 base64 chunks 一直挂在 `msgs` 里 |
| 后端另有 `patent_ids` SSE 事件，小程序也丢弃 | `general_agent.py:326-329` 发出；小程序 `handleEvent` 落 `default`（`chatStream.ts:310-311`），页面未注册 `onEvent` |

---

## 4. 数据层

### 4.1 内存 store

新增 `frontend/weapp/src/services/resultsStore.ts`，模块级 `Map<setId, ResultsPayload>`。

**理由**：小程序页面间不能传大对象（`navigateTo` 的 URL 参数很小），而 multi-MB 的结果集必须跨页可用。模块级可变状态在本仓库已有先例（`services/privacy.ts` 的 `pendingResolve` / `listener`）。

```ts
export interface ResultColumn { key: string; label: string; role: string }
export interface ResultsPayload {
  setId: string
  source: string
  columns: ResultColumn[]
  rows: Array<Record<string, string>>
}

/** 内存态：当前会话全量结果，按 setId 索引 */
export function putResults(p: ResultsPayload): void
export function getResults(setId: string): ResultsPayload | null
```

**写入时机**：对话页 `onArtifactsReady` 收到 `format === 'json'` 的工件时 → 逐 chunk 解 base64 → `JSON.parse` → `putResults`。解析失败**静默跳过**（对齐 web `chatSession.js:178-183`），不打断对话。

### 4.2 持久化

`Taro.setStorageSync('copiioai_results', { sets, index })`，**键名与结构与 web 同构**，便于将来对齐。

`index` 每项 `{ setId, sessionId, queryText, savedAt }`（对齐 `resultsStore.js:53-61`）。

**小程序存储比浏览器紧得多**：总量 10MB、**单键 1MB**（浏览器无单键限制）。故裁剪比 web 狠：

| 维度 | web | 小程序 | 理由 |
|---|---|---|---|
| 每集行数 | 50 | **40** | 留住 1MB 单键余量 |
| 摘要截断 | 500 字 | **400 字** | 摘要通常是最长字段 |
| 保留集数 | 100 | **20** | 10MB 总额，20 集 × ~40KB ≈ 800KB |
| index 条目 | 200 | **40** | 与集数同量级 |

- `url` role 的列**必须保留**——说明书 tab 要用
- 超限时丢最旧的集（**按 `savedAt` 排序**）；仍失败则静默放弃（对齐 web 的非配额错误处理）

  **此处有意偏离 web**：web 的 `dropOldestSet`（`resultsStore.js:38-43`）丢的是 `Object.keys(sets)[0]`，即**对象键序**而非 `savedAt`——而它的 `index` 是按插入序 unshift 维护的，两者语义并不一致（`resultsStore.js:53-61`）。那是 web 的潜在缺陷，不照搬。

### 4.3 一个必须讲清的取舍

裁剪**只作用于持久化的副本**。

- **当前会话**：内存里是**全量**——100 行就显示 100 行
- **重开小程序后**：从 storage 读回，**只剩前 40 行**

web 行为完全相同，不是本设计引入的。但它是用户可感知的差异，需在验收时明确。

---

## 5. 前端设计

### 5.1 文件清单

| 动作 | 文件 | 说明 |
|---|---|---|
| 新增 | `src/utils/results.ts` | **纯函数**：base64 逐 chunk 解码、按 role 取列、拼 meta 行、`pruneResults` 裁剪 |
| 新增 | `src/services/resultsStore.ts` | **只做 I/O 与状态**：内存 Map + `Taro.setStorageSync` 读写，内部调用上面的纯函数 |
| 新增 | `src/pages/results/index.tsx` | 列表页 |
| 新增 | `src/pages/results/index.config.ts` | 系统导航栏，标题「检索结果」 |
| 新增 | `src/pages/results/index.scss` | |
| 新增 | `src/components/ResultDetail/` | 页内详情覆盖层（三 tab） |
| 新增 | `src/services/patentDetail.ts` | spec / claims 调用 |
| 新增 | `src/utils/results.test.mjs` | 纯函数测试（见 §8） |
| 改 | `src/app.config.ts` | 注册 `pages/results/index` |
| 改 | `src/pages/chat/index.tsx` | 接 json 工件、结果入口、保存 `set_id` |
| 改 | `src/pages/chat/index.scss` | 入口按钮样式 |
| 改 | `package.json` | 加 `test` 脚本 |

### 5.2 入口

对话页助手消息下方，**只在有 json 工件时**出现：

```
[📄 查看全部 24 项结果]  [CSV] [XLSX] [MD]
```

- 没有 json 工件时**保持现状**（专利号 chip + 复制）——老会话不会冒出点不开的按钮
- 行数取 `rows.length`
- 点击 `Taro.navigateTo({ url: '/pages/results/index?set=' + setId })`

**入口的判定依据是消息上的 `resultSet`，不是 `artifacts` 里还有没有 json 项。** 解码入库后 json 工件会立刻从 `artifacts` 里移除（见 §9 内存风险），消息只留一个引用：

```ts
interface MsgView {
  // …既有字段
  /** 本轮结果集的引用。json 工件的原始 base64 解码入库后就不再留在消息里。 */
  resultSet?: { setId: string; rowCount: number }
}
```

入口渲染条件 = `m.resultSet` 存在。`m.resultSet.rowCount` 用于「查看全部 N 项结果」的 N。

**为什么保留旧的 chip 行**：`extractPatentIds` 是零后端依赖的兜底，且历史会话里已有它的位置。等到本功能稳定后可以再议是否撤掉，本轮不做。

### 5.3 历史会话回看：给消息加 `set_id`

保存时多带一个字段：

```ts
saveMessages(sid, [...history, userMsg, { role: 'assistant', content, set_id: setId }])
```

**为什么不学 web 按 `queryText` 文本匹配**：那是脆弱的相关性猜测。`set_id` 是确定的。

**为什么不需要改后端**：`PUT /session/{id}/messages` 对消息数组是**逐字透传**（`api_routes/session.py:202-224` 直接 `json.dumps`），且已有先例——`sources/long_task/task_messages.py` 就往消息里塞了 `patent_ids` / `meta` / `patent_data`。

重开会话时：消息带 `set_id` → 本地 storage 有对应结果 → 显示入口；没有（换设备/清缓存）→ 不显示。

**注意**：上传轮不调 `saveMessages`（后端拥有该轮消息，见 M3 spec §3.3），所以上传轮不会有 `set_id`。但上传轮产出的是长任务报告，不是结果列表，无影响。

### 5.4 列表页

- 路由 `pages/results/index?set=<setId>`
- **系统导航栏** `navigationBarTitleText: '检索结果'`——白拿原生返回键。对话页自绘 NavBar 是因为 ☰ 要占左上角，这里不需要
- 列表项对齐 web `ResultRow.tsx:31-65`：
  - **标题** = `title` role 列的值；无则 `—`
  - **meta 行** = 有值的 `patent_id` · `application_number` · `publication_number` · `assignee` · `publication_date`，用 ` · ` 连接
  - 行尾小箭头暗示可点
- **空态**：`getResults(setId)` 返回 null → 「结果已不可用，请重新检索」，**不是空白页**
- 顶部显示计数「共 N 项」
- **不做**排序/筛选/搜索/分页——对齐 web

### 5.5 详情面板（页内覆盖层）

页内覆盖，不跳新页：

- 顶部返回 + 三 tab：`详情` / `说明书` / `权利要求`
- **详情**：`title` 作标题，其余**全部非空**字段的键值表，标签用 `columns[].label`。数据全在已收到的 payload 里，**零新增请求**
- **说明书**：`GET /patent/{source}/{patentId}/spec` → `{success, pdf_url}` → `downloadFile` + `openDocument`（复用 `services/download.ts` 的 `downloadReport`/`openOrShareFile` 形态）
- **权利要求**：`GET /patent/{source}/{patentId}/claims` → `claims` 数组存在则逐项渲染 `number` + `独立`/`从属`（由 `independent` 布尔判定）+ `text`；否则回退 `pdf_url`

  **有意不渲染 `status` 字段**：其取值语义未核实（`api_routes/patent_detail.py` 未标注），不猜。若后续核实有意义（如「有效/失效」）再加。

**参数选择**：spec 用 `patentId || applicationNumber`；claims 用 `applicationNumber || patentId`；两者都传**行级** `source`。

**有意不做**：`doc` tab（web 用它 iframe 内嵌 PDF；小程序没有 iframe，且 `url` 列的 PDF 已由说明书 tab 覆盖）。

### 5.6 与既有代码的关系

- `utils/markdown.ts` 的 `MdTable` **不复用**——它是无 role 概念的扁平行表格，而结果列表需要「标题 + meta + 动作」的卡片形态
- 底部专利号 chip 行**保留**，仅在无 json 工件时出现

---

## 6. 错误处理

| 场景 | 行为 |
|---|---|
| 结果集本地找不到 | 列表页显示「结果已不可用，请重新检索」 |
| json 工件解析失败 | 静默跳过，不打断对话（对齐 web `chatSession.js:178-183`） |
| spec/claims 业务失败 | **按 `success:false` 判定，不能只看 HTTP 状态码**（`patent_detail.py:774-778`） |
| spec/claims 网络失败 | toast + 可重试，不影响其他 tab |
| PDF 打不开 | 复用 `openOrShareFile` 的 `shareFileMessage` 回退 |
| 存储写满 | 丢最旧的集；仍失败则静默放弃（对齐 web） |
| PR 无 `patent_id` 也无 `application_number` | 说明书/权利要求 tab 置灰并说明「该条缺少可查询的专利号」 |

---

## 7. 平台约束（沿用 M3 已确立的）

- **`wx.openDocument` 的 `fileType` 白名单没有 csv / md**——PDF 在白名单内，可用
- **小程序没有 `Blob` / `URL.createObjectURL` / `<a download>`**
- **SCSS 的 `px` 会被编译成 `rpx`；内联 style 必须写 `rpx`**
- **本仓库组件自 import 样式，页面与全局样式表中没有任何 `@import`**
- **服务端跑 Python 3.11**——本设计不改后端，但若改需守此线

---

## 8. 测试

小程序无测试框架，验证口径是 `tsc` + `build:weapp` + 真机人工验证。

**本轮新增**：给 `frontend/weapp` 加 `npm run test`（`node --test`），**只测纯函数**。

**为什么值得新增**：本功能的 bug 会藏在纯函数里，且**错了不报错，只会安静显示错数据**——

- `pruneResults`：裁剪后行数/摘要长度/集数是否正确，`url` 列是否被误删
- `decodeArtifactChunks`：**逐 chunk 独立解 base64 再拼字节**（拼字符串再解会在 256KB 边界上解错）
- `metaLine`：meta 只收录有值的 role、顺序正确、用 ` · ` 连接
- `pickColumn`：按 role 取值，无匹配时返回空

仓库已有先例：`frontend/nextjs/lib/results.test.mjs` 就是 `node --test` 的 `.mjs` 测试。

**其余**：`tsc` + `build:weapp` 双绿；真机走一遍验收标准。

---

## 9. 风险

| 风险 | 说明 | 处置 |
|---|---|---|
| 存储配额超限 | 单键 1MB / 总 10MB，比 web 紧 | §4.2 的裁剪；写入失败静默放弃，不影响对话 |
| spec/claims 接口慢或超时 | 后端要现场解析 USPTO 文档袋 | 按 tab 懒加载，不预取；失败可重试 |
| `source` 取值不匹配 | payload 的 `source` 可能是 `uspto_documents`，而详情接口的合法集是 `{uspto, google_patents, baiten}`（无 `uspto_documents`） | 文档行（`isDocument`）不显示说明书/权利要求 tab；行级 `source` 优先 |
| 内存不释放 | json 的 base64 chunks 目前一直挂在 `msgs` 里（multi-MB，且从不释放） | 解码入库后**立即从 `artifacts` 移除 json 项**，消息只留 `resultSet: {setId, rowCount}`（§5.2）。取出的引用是同一份数据，不与 store 重复持有 |
| 真机未验证 | 本仓库所有前端结论都只来自 `tsc` + build | 验收时必须真机走一遍 |

---

## 10. 验收标准

1. 回答下方出现「查看全部 N 项结果」入口，N 与真实行数一致
2. 无 json 工件的回答**不出现**该入口，仍显示原有专利号 chip
3. 进入结果页，列表显示标题 + meta 行（专利号 · 申请号 · 公开号 · 申请人 · 公开日，只列有值的）
4. 点任一行进入详情，看到全部非空字段的键值表，标签为中文列名
5. 说明书 tab 能拉取并打开 PDF
6. 权利要求 tab 结构化渲染；后端无结构化时回退 PDF
7. 返回键从详情回列表、从列表回对话页，行为符合小程序习惯
8. 重开小程序 → 历史会话 → 结果入口仍在，能打开且行数 ≤ 40
9. 换设备 / 清缓存后打开历史会话 → 不显示入口（而非显示点不开的入口）
10. `npm run test` 通过；`tsc` + `build:weapp` 双绿
11. 真机走一遍 1–9
