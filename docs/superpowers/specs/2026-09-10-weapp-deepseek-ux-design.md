# 小程序 DeepSeek 式交互重设计（形态一）+ 会话端点鉴权补全

- 日期：2026-09-10（brainstorm）／2026-09-11（定稿）
- 状态：设计已确认，待实现
- 分支：`feat/weapp`
- 前序：W3 微信登录（`0a3e95b`）、Taro M1 工程骨架（`e2a1b60`）、M2 流式对话页（`ffdb180`）、users 迁移（`b31d78c`）

---

## 1. 背景与问题

### 1.1 交互坏掉

现小程序是「会话列表页 → 对话页」两页结构：

- `pages/index/index.tsx` — 会话列表（首页），带右上角 `＋` 新建
- `pages/chat/index.tsx` — 对话页，靠 `useRouter().params.session_id` 从列表页跳入

列表空态文案在 `pages/index/index.tsx:85` 写的是「还没有对话，点击右下角开始」，但 `＋` 实际在**右上角**（`index.tsx:54`）。文案指向一个不存在的位置，新用户第一次进来就会卡住。

### 1.2 后端会话端点有 IDOR

`api_routes/session.py` 7 个端点中，5 个只靠 `session_id` 定位记录，不校验归属：

| 端点 | 行 | 现状 |
|---|---|---|
| `GET /session-by-id` | 64 | 无 `Authorization`，无归属校验 |
| `GET /session/{session_id}` | 98 | 无 `Authorization`，无归属校验 |
| `POST /session/{session_id}/message` | 153 | 无 `Authorization`，无归属校验 |
| `PUT /session/{session_id}/messages` | 182 | 有 token，**但没校验归属** |
| `DELETE /session/{session_id}` | 211 | 无 `Authorization`，无归属校验 |

`PUT` 那处尤其误导：`session.py:190` 的注释写着 "Verify session exists and belongs to user"，实际 SQL 只有 `WHERE session_id = %s AND status != 2`，没有 `user_id` 条件。注释与实现不符。

任何拿到（或猜到）`session_id` 的人都能读、改、删他人会话。

---

## 2. 目标与非目标

**目标**

1. 小程序改为 DeepSeek App 形态一：**首页即对话页**，左上角 `☰` 呼出抽屉收历史
2. 抽屉内支持**删除**与**重命名**
3. 补全 `api_routes/session.py` 全部会话端点的鉴权与归属校验

**非目标**

- 不做左缘滑手势呼出抽屉（与 `ScrollView` 手势冲突，实现量大）——已决
- 不做 M3（上传查重／分享／完整结果面板）
- 不改 web / nextjs 端的调用方式（见 §3.3 的兼容性核实）
- 不动 `PUT /messages` 的 `title` 字段语义（web 端在用，新端点与之并存）

---

## 3. 前端设计

### 3.1 页面结构变化

| 动作 | 文件 | 说明 |
|---|---|---|
| 改 | `frontend/weapp/src/app.config.ts` | `pages` 改为 `['pages/chat/index', 'pages/login/index']`，chat 成为首页 |
| 删 | `frontend/weapp/src/pages/index/` | `index.tsx` / `index.config.ts` / `index.scss` 三文件 |
| 改 | `frontend/weapp/src/pages/chat/index.tsx` | 挂 `NavBar` + `SessionDrawer` + 登录态检查 + 会话切换 |
| 改 | `frontend/weapp/src/pages/chat/index.config.ts` | 加 `navigationStyle: 'custom'`（见 §3.2） |
| 新增 | `frontend/weapp/src/components/NavBar/` | 自绘顶栏 |
| 新增 | `frontend/weapp/src/components/SessionDrawer/` | 会话抽屉 |
| 新增 | `frontend/weapp/src/components/RenameModal/` | 重命名弹窗 |
| 改 | `frontend/weapp/src/services/sessions.ts` | 加 `renameSession()` / `archiveSession()` |

`pages/index/index.tsx` 的既有职责全部搬家：

- 会话列表渲染 → `SessionDrawer`
- `useDidShow` 登录检查（`index.tsx:28-34`）→ chat 页（首页必须自己把门）
- `newChat()` / `openSession()` → chat 页内的会话切换逻辑

### 3.2 顶栏：`navigationStyle: 'custom'` 自绘

`☰` 要落在导航栏左上角，必须放弃系统导航栏，页面级配置 `navigationStyle: 'custom'`（只写在 `pages/chat/index.config.ts`，不影响 login 页）。

顶栏高度按微信官方公式算，避免与右上角胶囊按钮错位：

```
const menu = Taro.getMenuButtonBoundingClientRect()
const statusBarHeight = Taro.getSystemInfoSync().statusBarHeight
const navBarHeight = (menu.top - statusBarHeight) * 2 + menu.height
```

胶囊在**右上角**，`☰` 在左上角，水平方向不冲突；垂直方向按上式对齐即可。

`NavBar` 组件接口：

```ts
type Props = {
  title: string
  onMenuClick: () => void
}
```

标题取当前会话的 `title`；无当前会话或标题为空时显示「新对话」。因此 chat 页需把当前会话标题纳入 state——重命名当前会话后 `NavBar` 要跟着变。

### 3.3 抽屉

- 触发：`☰` 点击
- 动画：遮罩透明度淡入 + 面板 `transform: translateX(-100%) → 0`（只动 `transform`/`opacity`，不碰 `width`/`left`）
- 内容（自上而下）：`＋ 新对话` → 会话列表（`fetchSessions()`，开抽屉时拉取）
- 当前会话高亮
- 点会话条目 → 切换并关闭；点遮罩 / 再点 `☰` → 关闭
- **`＋ 新对话` 只重置本地状态**（清空 `msgs`、`sessionIdRef` 置空），**不立刻建会话**——会话仍在首条消息发出时创建（沿用 `pages/chat/index.tsx:117-121` 既有逻辑）。否则每点一次就留一条空会话

组件接口（展示型，不碰网络）：

```ts
type Props = {
  visible: boolean
  sessions: SessionItem[]
  currentSessionId: string
  loading?: boolean
  error?: string
  onClose: () => void
  onSelect: (sessionId: string) => void
  onNew: () => void
  onRename: (session: SessionItem) => void
  onDelete: (session: SessionItem) => void
}
```

### 3.4 删除与重命名

**平台约束**：微信小程序**没有带输入框的原生弹窗**。`Taro.showModal` 只有确定/取消，`window.prompt` 在小程序不存在。所以重命名必须自绘浮层 + `<Input>`，这就是 `RenameModal` 独立成组件的原因。

交互：

- 每条会话行右侧 `…` → `Taro.showActionSheet({ itemList: ['重命名', '删除'] })`
- **重命名** → `RenameModal`（预填当前标题）→ 确认 → `PUT /session/{id}/title`
- **删除** → `Taro.showModal` 二次确认 → `DELETE /session/{id}`（后端归档 `status=2`）
- **删的正好是当前打开的会话** → 清空对话回空态，避免停在一个已归档会话上

`RenameModal` 接口（展示型，不碰网络）：

```ts
type Props = {
  visible: boolean
  initialTitle: string
  busy?: boolean
  error?: string
  onCancel: () => void
  onConfirm: (title: string) => void
}
```

职责划分：**抽屉和弹窗是纯展示组件**（只发意图），**chat 页持有状态与网络调用**（拉列表、调接口、刷新、错误处理）。

---

## 4. 后端设计

### 4.1 新增端点

```
PUT /session/{session_id}/title
  body: { title: str }
  鉴权 + 归属校验 → 更新 conversations.title
```

用专用端点而非复用 `PUT /messages`，是因为后者会**重写整个 messages 数组**；流式对话进行中改名会产生丢消息竞态。

### 4.2 鉴权／归属补全

统一模式：

```python
user = verify_firebase_token(http_request.headers.get("Authorization"))
user_id = int(user['uid'])
```

归属查询统一为 `WHERE session_id = %s AND user_id = %s AND status != 2`，**查不中返回 404**（而非 403）——不暴露"会话存在但不属于你"。

`verify_firebase_token`（`sources/user/passport.py:31`）在 header 缺失时抛 `HTTPException(401, "Missing token")`；wx_ 分支（`:45-54`）返回 `{"uid": int(user_id), ...}`，与 Firebase 分支同一出口，故 `int(user['uid'])` 对两端通用。

### 4.3 兼容性核实（加鉴权不会打断线上）

已逐一核对全部 8 个生产调用点，**均带 `Authorization: Bearer`**：

| 端点 | 调用方 | 凭证来源 |
|---|---|---|
| `GET /session-by-id` | `frontend/nextjs/services/api.ts:190` | `authHeaders()`（`:188`） |
| `GET /session/{id}` | `frontend/web-app/src/services/sessionService.js:38` | `get()` → `authHeaders()` |
| `GET /session/{id}` | `frontend/weapp/src/services/chat.ts:37` | `request()` 自动注入 |
| `POST /session/{id}/message` | `frontend/web-app/src/services/sessionService.js:44` | `post()` → `authHeaders()` |
| `PUT /session/{id}/messages` | `frontend/nextjs/services/api.ts:202` | `authHeaders()`（`:201`） |
| `PUT /session/{id}/messages` | `frontend/nextjs/contexts/ChatContext.tsx:192`（pagehide keepalive） | `authHeaders()`（`:190`） |
| `PUT /session/{id}/messages` | `frontend/weapp/src/services/chat.ts:47` | `request()` |
| `DELETE /session/{id}` | `frontend/web-app/src/services/sessionService.js:47` | `awaitAuthHeaders()`（`:49`） |

结论：本次加鉴权对既有前端是**纯收紧，不产生回归**。

---

## 5. 错误处理

| 场景 | 行为 |
|---|---|
| 401 未登录/凭证失效 | `services/api.ts:36` 已拦截：清 `wxToken`/`userId` + 跳登录页 |
| 404 会话不存在或非本人 | 抽屉内提示「会话不存在或已删除」并刷新列表 |
| 重命名失败 | 弹窗不关闭，显示错误文案，可重试 |
| 列表拉取失败 | 抽屉内显示错误（复用 `errorText()`），不阻塞对话 |

---

## 6. 测试

`tests/test_session_api.py`（已存在）补充：

- **归属校验**：用户 A 无法 `GET` / `PUT` / `DELETE` 用户 B 的会话 → 均 404
- **鉴权**：无 `Authorization` → 401（覆盖 5 个补全端点）
- **`PUT /title`**：正常路径；空标题 / 超长标题边界
- **回归**：`PUT /messages` 带 `title` 字段的既有行为不变

前端：`npm --prefix frontend/weapp run tsc` + `npm run build:weapp` 双绿（沿用 M1/M2 的验证口径）。

---

## 7. 风险

| 风险 | 说明 | 处置 |
|---|---|---|
| 自绘顶栏在真机上的状态栏/胶囊错位 | 公式在模拟器与真机可能不一致 | 微信开发者工具 + 真机各看一次 |
| 抽屉遮罩与小程序的 `catchMove` 滚动穿透 | 遮罩打开时底层 ScrollView 仍可滚动 | 遮罩节点绑 `catchMove` |
| 删除当前会话后的状态清理遗漏 | 会停在已归档会话上，后续发消息 404 | 显式测这条路径 |

---

## 8. 验收标准

1. 小程序打开即是对话页，左上角有 `☰`
2. `☰` 呼出抽屉，列出历史会话，当前会话高亮
3. 抽屉内可新建、切换、重命名、删除会话
4. 删除当前会话后回到空态，可正常开始新对话
5. 用另一账号的 token 请求他人 `session_id` → 404
6. 线上 web / nextjs 端会话功能回归正常
