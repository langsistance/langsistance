# 聊天页空状态落地布局设计（Chat Landing）

日期：2026-08-18
状态：已获用户批准（分三节确认：布局交互 / 六大能力文案 / 视觉样式）

## 背景

当前 `/app/chat`（登录后进入）空状态只有欢迎语 + SceneHint，输入框固定在底部。目标：像 DeepSeek / ChatGPT 落地页一样，无消息时把输入框放到屏幕中间，slogan 在上、六大能力卡片在下；用户提问后输入框回到底部。

范围：**仅聊天页空状态**。营销落地页 `/`、结果页 SceneHint、web-app（Vite）均不改动。

## 布局结构（空状态）

```
            （整体垂直居中）
   ┌────────────────────────────────┐
   │    「专利情报，一问即得」           │  ← slogan（h2，居中）
   │   ┌──────────────────────────┐ │
   │   │  [📎] 输入框……  [发送]    │ │  ← 居中输入框（复用 chat-input-wrapper 样式）
   │   └──────────────────────────┘ │
   │   ┌──────┐ ┌──────┐ ┌──────┐  │
   │   │ ①    │ │ ②    │ │ ③    │  │  ← 六大能力（3列×2行）
   │   │ ④    │ │ ⑤    │ │ ⑥    │  │
   │   └──────┘ └──────┘ └──────┘  │
   │   ── 当前可用能力（SceneHint）──  │
   └────────────────────────────────┘
```

### 交互行为

1. `messages.length === 0` 时渲染新 `ChatLanding` 组件：slogan + 居中输入框 + 六大能力 + SceneHint，整体垂直居中于聊天区域。
2. 用户发送第一条消息 → 自动切回现有聊天视图（消息列表 + 底部输入框）。无动画，直接切换。
3. 输入区（文件附件按钮 + textarea + 发送/停止按钮 + 文件 chips 条 + 拖拽/粘贴处理）提取为可复用组件 `ChatComposer`，`ChatLanding` 与现有聊天视图共用，输入状态（`ChatContext` 的 `input`/`setInput`、`useChatStream` 的 `send`/`abort`/`selectedFiles` 等）不变。
4. `SceneHint` 从消息区顶部移到六大能力下方，上方加小节标题「当前可用能力 / Currently available」。数据仍走后端动态接口（已启用场景 + 智能问答/深度研究），结果页的 SceneHint 调用不受影响。
5. 空状态期间的拖拽/粘贴文件上传、Enter 发送等现有行为全部保留。

### 响应式

- 桌面：六大能力 3 列 × 2 行
- 平板（<1024px）：2 列
- 手机（<640px）：1 列

## 六大能力卡片内容（中英双语）

纯展示，不可点击。无图标，文字卡片：标题（粗体）+ 描述（次级色）。

| # | 标题（中） | 标题（EN） | 描述（中） | 描述（EN） |
|---|---|---|---|---|
| ① | 美国专利检索 | US Patent Search | 直达 USPTO，按专利号、申请人、技术关键词精准检索美国专利 | Search USPTO directly by patent number, assignee, or technical keywords |
| ② | 自然语言检索 | Natural Language Search | 用大白话提问，AI 自动理解技术方案，无需检索式、无需 IPC 分类号 | Ask in plain language — AI understands your intent, no query syntax or IPC classes needed |
| ③ | 审查历史分析 | Prosecution History Analysis | 自动梳理 OA 答复、修改记录与关键争辩点，生成时间线与风险摘要 | Auto-trace office actions, amendments and key arguments into a timeline with risk summary |
| ④ | 跨国同族专利审查历史分析 | Cross-Border Family Prosecution Analysis | 透视同族专利全球布局与各国审查历程，多国状态一目了然 | See family global layout and per-country prosecution status in one view |
| ⑤ | 全流程文件下载 | Full-Cycle File Downloads | 专利说明书、权利要求、分析报告，PDF / DOCX 一键下载 | Download patent specs, claims and AI analysis reports as PDF / DOCX |
| ⑥ | 免费 | Free | 六大能力当前全部免费开放，无需信用卡，无隐藏费用 | All six capabilities are currently free — no credit card, no hidden fees |

### 免费措辞约束

- 第 ⑥ 张只说「免费 / Free」，**不得出现「永久免费」**。文案使用「当前全部免费开放 / currently free」。
- 其他五张卡片**不加** FREE 徽章（免费信息由第 ⑥ 张承担）。

### Slogan

- 中：专利情报，一问即得
- EN：Patent intelligence, just ask.

## 视觉样式

- 沿用现有 app 设计语言（CSS 变量体系：`--color-bg-main`、`--color-bg-white`、`--color-text-primary/secondary`、`--color-primary`、`--color-border`），不引入参考页的薄荷绿第二套配色。
- 背景：沿用聊天页现有背景，不加渐变块。
- Slogan：`text-3xl md:text-4xl` 粗体，`--color-text-primary`，居中，与输入框间距约 20–24px。
- 输入框：复用现有 `chat-input-wrapper`（圆角 24px、focus 主色描边），720px 上限居中。
- 六大能力卡片：白底 + 细边框 + 圆角 16px；标题 16px 粗体；描述 13px 次级色；hover 轻微上浮 + 边框变色（用 app 主色）。无图标。
- 第 ⑥ 张「免费」卡片：用强调色（琥珀/橙色系）文字或边框与其余五张区分。
- 「当前可用能力」小节：复用现有 `scene-hint` 样式，仅在上方加小节标题（`text-sm` 次级色）。
- 动效：仅 hover 过渡 + 首屏 `fadeIn`（复用现有 `fadeIn 0.3s`），尊重 `prefers-reduced-motion`。
- 无障碍：slogan 用 `h2`；卡片为纯展示 `div`，无交互语义。

## 实现要点（文件级）

| 文件 | 改动 |
|---|---|
| `app/(auth)/chat/page.tsx` | 空状态改为渲染 `<ChatLanding />`；提取输入区为 `ChatComposer` 并共用；`SceneHint` 移入 `ChatLanding` |
| `components/app/ChatComposer.tsx`（新建） | 输入区组件：文件 chips、附件按钮、textarea、发送/停止、拖拽粘贴逻辑（自 `page.tsx` 提取） |
| `components/app/ChatLanding.tsx`（新建） | 空状态落地视图：slogan + `ChatComposer` + 六大能力网格 + 小节标题 + `SceneHint` |
| `styles/popup.css` | 新增 `chat-landing` / 六大能力卡片 / 小节标题样式 |
| `lib/app-i18n/locales/zh.ts`、`en.ts` | 新增 `chat.landing` 命名空间：slogan、六大能力 6×2 文案、小节标题 |

### 样式命名约定

- 容器：`.chat-landing`
- slogan：`.chat-landing-slogan`
- 能力网格：`.chat-landing-grid`，卡片 `.chat-landing-card`（`.free` 修饰免费卡）
- 小节标题：`.chat-landing-section-title`

## 验证

- `npm run build`（nextjs）通过
- 手动验证：未提问时输入框居中 + slogan + 6 卡片 + 可用能力；提问后回到底部布局；zh/en 切换后全部文案正确；拖拽/粘贴文件、Enter 发送在空状态下正常；结果页 SceneHint 不受影响
