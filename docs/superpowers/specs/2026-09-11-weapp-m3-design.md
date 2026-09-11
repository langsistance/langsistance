# 小程序 M3 收尾包：文件上传下载 + 分享 + 隐私政策（附长任务端点越权修复）

- 日期：2026-09-11（brainstorm 定稿）
- 状态：设计已确认，待实现
- 分支：`feat/weapp`
- 前序：形态一 spec（`2026-09-10-weapp-deepseek-ux-design.md`，已交付）、M2 流式对话页
- 依据：`deploy/china/MIGRATION_PLAN.md:142` 的 M3 定义（上传查重 / 历史会话 / 分享 / 隐私政策，出口「可提审」）

---

## 1. 范围

M3 在迁移计划里是一行，拆开是六件事。本轮做四件 + 一件后端修复：

| # | 事项 | 本轮 | 说明 |
|---|---|---|---|
| 1 | 文件上传（PDF/DOCX/XML） | ✅ | **单文件**，见 §3.2 |
| 2 | 文件下载（CSV/XLSX 工件 + docx/pdf 报告） | ✅ | 两条链路，落地方案不同，见 §3.4 |
| 3 | 长任务进度 | ✅ | 上传后必需，见 §3.3 |
| 4 | 分享 | ✅ | 仅转发卡片，见 §5 |
| 5 | 隐私政策 | ✅ | 页面 + 公众平台配置，见 §6 |
| 6 | 长任务端点越权修复 | ✅ | 顺带安全修复，见 §4 |
| — | 历史会话 | 已完成 | 形态一 spec 已交付 |
| — | 外观比对 / 图片上传（seller-scene） | ❌ | 能力全在 `feat/seller-scene`（52 提交未合） |
| — | 一次上传多文件 | ❌ | 平台硬约束，见 §2.1 |
| — | 分享特定会话/报告给外部人 | ❌ | 与 §4 的权限收紧冲突，见 §5 |

---

## 2. 平台约束（本轮设计的边界）

三条经文档与类型定义核实的约束，决定了后面所有设计。**这三条推翻了若干"照抄 web"的直觉**。

### 2.1 `wx.uploadFile` 只能单文件

- Taro 类型：`frontend/weapp/node_modules/@tarojs/taro/types/api/network/upload.d.ts:5-35` 的 `Option` 只声明 `filePath`(string) + `name`(string)，无 `files`
- 微信原生 `wx.uploadFile` 参数表同样只有 `filePath`(必填) + `name`(必填)
- **`files` 数组参数是 uni-app 的扩展，微信原生没有**；`filePath` 传数组报 `parameter.filePath should be String instead of Array`
- 后果：web 的 `<input multiple>` 一次传 N 个文件汇成一份报告的语义，小程序**无法复刻**。循环上传会得到 N 个独立任务 N 份报告——语义不同，且会撞后端并发限制（`api_routes/core.py:947-950` 返回 429）

### 2.2 `wx.uploadFile` 没有 `onChunkReceived`

- `UploadTask` 只有 `abort` / `onProgressUpdate` / `onHeadersReceived`；`onChunkReceived` 是 `RequestTask` 的方法
- 但**上传分支不需要流式**：`POST /query_stream` 的 multipart 分支只回 `long_task_created` + `end` 两帧（`api_routes/core.py:1208-1219`），`res.data` 直接拿到完整文本

### 2.3 `wx.openDocument` 不支持 csv

- `fileType` 合法值：`doc` / `docx` / `xls` / `xlsx` / `ppt` / `pptx` / `pdf` —— **无 csv**
- 另：小程序没有 `Blob` / `URL.createObjectURL` / `<a download>`，web 的下载实现（`MarkdownMessage.tsx:100-110`、`:225-237`）**整体不可移植**

---

## 3. 前端设计（`frontend/weapp`）

### 3.1 文件清单

| 动作 | 文件 | 说明 |
|---|---|---|
| 改 | `src/services/chatStream.ts` | 导出 `SseParser`；补 `artifact_*` 帧处理；`push_filter` 对齐 |
| 改 | `src/services/api.ts` | 抽出 401 处理为可复用导出（`uploadFile` 走不到 `request()`） |
| 新增 | `src/services/upload.ts` | `chooseMessageFile` + `uploadFile` + 解析 `long_task_created` |
| 新增 | `src/services/longTask.ts` | 轮询、报告下载、重试 |
| 新增 | `src/services/download.ts` | 临时文件落盘 + `openDocument` / `shareFileMessage` |
| 新增 | `src/components/AttachmentBar/` | 输入栏上方的附件 chip 条 |
| 新增 | `src/components/LongTaskCard/` | 长任务进度卡片 |
| 新增 | `src/components/PrivacyPopup/` | 隐私授权浮层（见 §6.3，上传的硬前置） |
| 新增 | `src/services/privacy.ts` | `onNeedPrivacyAuthorization` 注册与 resolve 句柄管理 |
| 改 | `src/app.ts` | App 级注册隐私授权监听（一次，非页面级） |
| 改 | `src/pages/chat/index.tsx` | 接 `onEvent`、附件状态、任务状态、下载入口 |
| 改 | `src/pages/chat/index.scss` | 上述组件样式 |
| 新增 | `src/pages/privacy/` | 隐私政策页（`index.tsx` / `index.config.ts` / `index.scss`） |
| 改 | `src/app.config.ts` | `pages` 加 `pages/privacy/index` |
| 改 | `src/pages/login/index.tsx` | 隐私政策可达入口 |

### 3.2 上传

**选文件**

```ts
Taro.chooseMessageFile({ count: 1, type: 'file', extension: ['pdf', 'docx', 'xml'] })
```

`extension` 仅在 `type: 'file'` 时生效。返回 `res.tempFiles[0]` = `{ path, name, size, type, time }`。

**校验**（对齐 web 的阈值，但**不静默丢弃**）

web 在 `frontend/nextjs/lib/useChatStream.ts:74-91` 对超限/不支持的文件是 `continue` 静默跳过，用户不知道文件为什么没进去。小程序改为 `Taro.showToast` 明确提示：

| 条件 | 提示 |
|---|---|
| `size > 10MB` | 文件超过 10MB 上限 |
| `size < 50B` | 文件过小，可能不是有效文档 |
| 扩展名不在白名单 | 仅支持 PDF / DOCX / XML |

阈值与 web 一致：`MAX_FILE_SIZE = 10 * 1024 * 1024`、`MAX_FILE_COUNT = 100`（小程序单文件，仅用前者），后端同值见 `api_routes/core.py:953-954`。

**上传**

```ts
Taro.uploadFile({
  url: `${API_BASE}/query_stream`,
  filePath: file.path,
  name: 'patent_files',
  formData: {
    query,                                  // 必填，后端 core.py:935 为空直接 400
    query_id: queryId,
    push_filter: '2',                       // 见下方「既有不一致」
    conversation_history: JSON.stringify(history),
    session_id: sid,                        // 有会话时才带
  },
  header: { Authorization: `Bearer ${token}` },
  timeout: 120000,
})
```

字段名与 web 完全一致（`frontend/nextjs/services/api.ts:139-146`）。

**响应解析**：`res.data` 是 SSE 文本 → `new SseParser().push(res.data)` 得到事件数组。因此 `SseParser` 需从 `chatStream.ts` 导出（当前是文件内私有）。

**⚠️ 既有不一致（本轮一并处理）**：web 两条路径都发 `push_filter: 2`（`frontend/nextjs/services/api.ts:117` 数字、`:141` 字符串），**小程序 JSON 路径发的是 `null`**（`src/services/chatStream.ts:148`）。后端 `sources/knowledge/query_filters.py:74-76` 收到 `None` 直接返回空条件——**等于小程序现在完全没走知识库推送过滤**。本轮把两条路径统一到 `2`：JSON 路径发数字 `2`（照 web `:117`），multipart 的 `formData` 值只能是字符串，发 `'2'`（照 web `:141`）。

**发送按钮禁用条件 = `!input.trim()`**（有文件但无文字也禁用）。
理由：后端 `api_routes/core.py:935` 要求 `query` 非空。web 的 `ChatComposer.tsx:165` 此时按钮**可点**，但 `useChatStream.ts:119` 的 `if (!text || streaming) return` 让点击静默无效——这是 web 的 bug，不照抄。

### 3.3 长任务

上传后到出报告要数分钟，必须有进度呈现。小程序目前**完全没有长任务 UI**。

**数据流**：收到 `long_task_created`（`{ task_id, session_id, patent_ids, patent_count, source, status }`）→ 挂到 assistant 消息 → 每 1.5s 轮询 `POST /long_task/batch_status`（web 同节奏，`frontend/nextjs/lib/useChatStream.ts:629-630`）→ 终态停。

**消息结构**（`MsgView` 增字段）：

```ts
task?: {
  id: string
  status: 'pending' | 'queued' | 'running' | 'paused' | 'cancelling'
        | 'completed' | 'failed' | 'cancelled' | 'unknown'
  // 'unknown' 两个来源：轮询连续失败（见 §7）；后端 get_task_status 对
  // Redis 记录已过期的任务也返回 'unknown'（status_manager.py:112-117）
  phase?: string
  progress?: number
  step?: string
  reportFiles?: Array<{ format: string; filename: string; size: number }>
  error?: string
}
```

**与 web 的实现差异（有意为之）**：web 把任务状态编码进消息正文，再用正则从文本反解（`frontend/nextjs/components/app/LongTaskProgress.tsx:43-135`，靠 `🔬`/`✅`/`❌`/`[N%]` 标记）。那是为了绕过"消息体只能是字符串"的限制。小程序的消息是结构化对象，直接挂字段，**不做标记编码**。这是实现差异，不是产品差异。

**阶段文案**对齐 web 的 `STANDARD_PHASES`（`LongTaskProgress.tsx:224-231`）：`extracting_text` / `searching_patents` / `generating_columns` / `analyzing` / `generating_report` / `exporting`。

**轮询生命周期**：页面级定时器。`useDidHide` 停止，`useDidShow` 对未完成任务恢复。不用 web 的全局单定时器（`useChatStream.ts:525-632`）——小程序页面栈浅，页面级足够。

**历史恢复**：后端在任务完成时通过 `append_task_message` 把结果写进 `conversations.messages`（`sources/long_task/status_manager.py:162`）。所以重开历史会话看到的是后端写好的完成态文本，不是小程序的任务卡片。与 web 行为一致。

**重试**：`failed` 态显示错误 + 重试按钮 → `POST /long_task/{task_id}/retry`（`api_routes/long_task.py:426-541`）。

### 3.4 下载

两条来源链路，落地方式不同：

| 来源 | 传输形态 | 落地 |
|---|---|---|
| 检索工件 CSV/XLSX | base64 分片（`artifact_start`/`chunk`/`end`） | `FileSystemManager.writeFile({ encoding: 'base64' })` |
| 长任务报告 docx/pdf | 二进制 | `Taro.downloadFile` → `tempFilePath` |

**A. 接上被丢弃的工件帧**

现在这些数据**全被丢掉**：`src/services/chatStream.ts:20` 定义了 `onEvent` 回调，但 `src/pages/chat/index.tsx:274-292` 从没接上，`artifact_*` 帧全部落进 `default` 分支静默丢弃（`chatStream.ts:221-222`）。

处理方式照 web 的关键决策（`frontend/nextjs/lib/useChatStream.ts:270-279`）：

- `artifact_start` → 建缓冲 `{ artifactId: { filename, mimeType, rowCount, columnCount, chunks: [] } }`
- `artifact_chunk` → `chunks.push(data)`，**不 setState**
- `artifact_end` → 一次性 setState 挂到消息上

**「不 setState」是硬要求**：web 的注释（`useChatStream.ts:275-276`）明确记录，每个 chunk 都重建数组会让前端卡死数分钟（工件是 multi-MB 级，`sources/callback/sse_callback.py:8-10`）。

**B. 打开/转发**

| 格式 | 动作 |
|---|---|
| `xlsx` / `pdf` / `docx` | `Taro.openDocument` 预览 |
| `csv` | `Taro.shareFileMessage` 转发到聊天（`openDocument` 不支持 csv） |
| 消息正文 `.md` | `shareFileMessage`（.md 同样不在 `openDocument` 白名单） |

「消息正文导出为 Markdown」对齐 web 的 `MarkdownMessage.tsx:210-223`（web 文件名 `CopiioAI_Chat_<时间戳>.md`）。

**C. 临时文件清理**

`USER_DATA_PATH` 有 200MB 上限。工件用 `writeFile` 写入时路径由我们决定，因此把**上次写入的路径记在模块级变量**，本次写盘前先 `unlink` 它——只留最新一个，不做目录扫描。
（`Taro.downloadFile` 落到的是系统临时目录，不受此限，无需清理。）

---

## 4. 后端设计：长任务端点越权修复

M3 正是要把下载能力暴露到小程序新入口，此时不修等于把洞接到新入口上。

现状（`api_routes/long_task.py`）：

| 端点 | 行 | 鉴权 | 归属校验 |
|---|---|---|---|
| `GET /long_task/{task_id}/report` | `:633-668` | ❌ 无 | ❌ 无 |
| `GET /long_task/{task_id}/status` | `:552-559` | ✅ 有 token | ❌ 无 |
| `POST /long_task/batch_status` | `:561-575` | ✅ 有 token | ❌ 无 |

`task_id` 形如 `lt_ + uuid4().hex[:12]`（`api_routes/core.py:984`）。report 端点**不带任何凭证即可下载他人报告**。

修复沿用 `api_routes/session.py` 那轮的约定：

- **report**：加 `verify_firebase_token` + 归属查询 → 非本人或不存在的 `task_id` **一律 404**（不暴露"存在但不属于你"）
- **status**：同上，非本人 404
- **batch_status**：一条 `SELECT task_id FROM long_tasks WHERE task_id IN (...) AND user_id = %s`，**只返回属于本人的**，不在结果里的不返回（不报错，避免被用来枚举 task_id 存在性）

`long_tasks` 表已有 `user_id` 列，`sources/long_task/status_manager.py:369` 已在查它，无需改表结构。

**兼容性核实（实现前必做）**：web 端在调这三个端点（`frontend/nextjs/services/api.ts:222-291`、`frontend/nextjs/lib/longTaskRecovery.ts:9-32`）。需逐一确认 web 轮询的**都是自己创建的任务**；若存在跨用户查看的既有用法，先停下来讨论。加校验后必须跑 web 回归。

---

## 5. 分享

**先说清落差：web 上没有"分享对话/报告"的功能。** web 的 `share` 页（`frontend/nextjs/app/app/(auth)/share/page.tsx`）是**知识库条目在用户间授权共享**，与小程序聊天无关。这块没有 web 可参照，是纯小程序原生能力。

本轮实现转发卡片：

- `Taro.useShareAppMessage` → 转发给好友，卡片标题取当前提问/会话标题，`path` 指向首页
- `Taro.useShareTimeline` → 分享到朋友圈
- `Taro.showShareMenu({ menus: ['shareAppMessage', 'shareTimeline'] })`

**为什么只能做到这一步**：分享**具体某个会话**要求接收方打得开，而会话刚做完 IDOR 归属校验（非本人一律 404）——转发出去对方只会看到"会话不存在"。要做"分享报告给外部人看"必须新建公开分享链路（带 token 的只读页），那是独立一轮的活，且与刚收紧的权限模型正面冲突。**本轮不做。**

提审角度也够用：微信不要求分享特定内容，只要求分享行为合规（不诱导分享）。

---

## 6. 隐私政策

分两截，**一截是代码、一截是后台配置，必须都做**：

1. **小程序内页面**（代码）：新增 `pages/privacy/index`，中文正文。
   现有 web 隐私政策**不能直接用**——`frontend/nextjs/app/(landing)/privacy-policy/page.tsx` 是英文、写的是 Chrome 扩展权限（`storage`/`tabs`/`webRequest`/`scripting`/`identity`/`offscreen`/`cookies`）、最后更新 2026-03-05，语境完全不符。需针对小程序重写。
2. **微信公众平台《小程序用户隐私保护指引》**（后台配置，非代码）：声明收集哪些信息。
   有利条件：本项目登录只走 `wx.login` 拿 code 换 openid（`sources/user/wechat_login.py`），**不取昵称/头像/手机号**，声明可以做得很轻。
3. **隐私授权弹窗**（代码）——**这条是上传功能的硬前置，不是可选项**：
   `wx.chooseMessageFile` **在微信隐私接口清单内**（对应声明的信息类型是「收集你选中的文件」）。规则是：**未在《小程序用户隐私保护指引》中声明该信息类型，接口直接被禁用**——不是调用失败，是压根调不起来。
   因此实现需要一个弹窗组件 + 授权流程：
   - App 级监听 `wx.onNeedPrivacyAuthorization(resolve => {...})`，拿到 `resolve` 后弹自定义浮层
   - 浮层内用 `<button open-type="agreePrivacyAuthorization">` 让用户同意，回调里 `resolve({ buttonId, event: 'agree' })`
   - `wx.requirePrivacyAuthorize`（基础库 2.32.3+）是**辅助**接口，官方明确"不是必须调用"——用它可以在真正调 `chooseMessageFile` 之前提前触发弹窗，避免用户选完文件才发现要授权。本轮采用这个更顺的顺序。

   顺序影响体验，务必按此实现：**先授权 → 再选文件**。否则用户点回形针 → 选完文件 → 才弹协议，白选一次。

---

## 7. 错误处理

| 场景 | 行为 |
|---|---|
| 上传 401 | 复用 `api.ts` 既有逻辑：清 `wxToken`/`userId` + 跳登录页 |
| 上传 429（日限/并发） | 显示后端返回的文案，不重试 |
| 上传成功但任务 `failed` | 卡片显示错误 + 重试按钮 |
| 轮询网络失败 | 不立即结束任务；**连续 5 次失败**（约 7.5s）才置 `status: 'unknown'` 并停止轮询，卡片显示"状态获取失败，稍后重试" |
| 报告 404 | 提示"报告不存在或已过期" |
| `openDocument` 失败 | 回退到 `shareFileMessage` 转发 |
| 文件超限/类型不支持 | `showToast` 明确提示（不学 web 的静默丢弃） |
| 写盘失败（`USER_DATA_PATH` 满） | 清理历史临时文件后重试一次，仍失败则提示 |

---

## 8. 测试

**后端**（`tests/test_long_task_api.py` 已存在，补充）：

- **归属校验**：用户 A 查/下载用户 B 的 `task_id` → 均 404
- **鉴权**：`/report` 无 `Authorization` → 401
- **`batch_status`**：混杂本人与他人的 task_ids → 只返回本人的，且不报错
- **回归**：本人任务的状态轮询与报告下载行为不变
- **web 回归**：确认 nextjs 的三个调用点在加校验后仍正常

测试断言按 session 那轮的教训，**断言 SQL 文本与实际绑定参数**，不只断言状态码——否则"参数没传对"也能过。

**前端**：`npm --prefix frontend/weapp run tsc` + `npm run build:weapp` 双绿（沿用 M1/M2 的验证口径）。小程序无测试框架。

**真机验证（本轮必须做一次）**：开发者工具与真机在文件系统、`openDocument`、`shareFileMessage` 上行为不同。历史教训：排版环节所有结论都只在 Windows 开发者工具上看过，真机观感不同（见 memory `weapp-devtools-pitfalls`）。

---

## 9. 风险

| 风险 | 说明 | 处置 |
|---|---|---|
| 加归属校验打断 web | nextjs 有三个调用点 | 实现前逐一核实调用方；跑 web 回归 |
| 公众平台隐私指引未声明 → `chooseMessageFile` **直接被禁用** | 后台配置是代码之外的一步，最容易漏；且此时代码里的授权弹窗也不会生效，真机自测会卡在"选文件没反应" | 提审前先在公众平台声明「收集你选中的文件」；把这一步写进 `deploy/china/RUNBOOK_GO_LIVE.md` 的上线清单 |
| 大文件上传超时 | 10MB 在弱网下可能超过 120s | 已设 `timeout: 120000`；用 `onProgressUpdate` 给进度反馈 |
| base64 工件内存占用 | multi-MB 工件在 JS 侧以 base64 字符串累积 | 照 web 的"chunk 不 setState"决策；`artifact_end` 一次性落盘后释放 |
| `USER_DATA_PATH` 200MB 上限 | 反复下载会堆积 | §3.4-C 的前缀清理 |
| 临时文件在真机被系统回收 | 下载与打开之间存在时间差 | 打开失败时重新下载一次 |

---

## 10. 验收标准

1. 输入栏可选取 PDF/DOCX/XML（单个），选中后显示文件名/大小/移除
2. 发送后显示长任务进度卡片，阶段与百分比随轮询更新
3. 任务完成后可下载 docx 与 pdf 报告，能在小程序内打开
4. 检索类回答的 xlsx 可打开、csv 可转发到聊天
5. 消息正文可导出为 Markdown 并转发
6. 有文件但无文字时发送按钮禁用（不出现点击无效）
7. 转发卡片标题带当前提问/会话标题
8. 小程序内可访问中文隐私政策页
8b. 首次点回形针时先弹隐私授权，同意后才进入选文件（不是选完才弹）
8c. 公众平台已声明「收集你选中的文件」，`chooseMessageFile` 真机可正常唤起
9. 用另一账号的 token 请求他人 `task_id` 的状态与报告 → 均 404
10. web 端长任务轮询与报告下载回归正常
11. `tsc` 与 `build:weapp` 双绿
