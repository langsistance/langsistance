# CopiioAI 信息安全声明

- 日期：2026-09-16（brainstorm 定稿）
- 状态：设计已确认，待实现
- 分支：`feat/security-statement`（自 `main` @ `4db28ef` 切出）
- 目标：在用户即将输入专有技术方案的位置，给出三天可核实的保密承诺

---

## 1. 范围

| # | 事项 | 本轮 | 说明 |
|---|---|---|---|
| 1 | 应用首屏信任模块 | ✅ | 主战场，见 §3.2 |
| 2 | 登录弹窗信任行 | ✅ | 见 §3.4 |
| 3 | 输入框聚焦轻提示 | ✅ | 见 §3.5 |
| 4 | `/security` 信息安全页 | ✅ | 新建，见 §4 |
| 5 | `/privacy-policy` 重写 | ✅ | 现内容为旧产品线口径，见 §5 |
| 6 | 后端底座加固（日志脱敏 / 过期删除 / 审计） | ❌ | 独立工作包 W2–W3，见 §7 |
| 7 | 认证徽章（等保 / ISO / SOC 2） | ❌ | 未持有，不得展示，见 §2.2 |
| 8 | 子处理者清单（含第三方厂名） | ❌ | 与 `no-vendor-name-disclosure` 记忆冲突，且用户已选择不披露处理位置 |

---

## 2. 为什么这样写

### 2.1 写作原则

**只写做得到的事，并为后续升级预留位置。**

声明的可信度不来自措辞的强度，来自每一句都能被代码验证。用户是专利代理人、企业 IP 负责人、研发工程师——他们中相当一部分会带着尽调的心态读这段话。任何一句被戳破，整份声明的可信度归零，且会污染产品整体的可信度。

因此本轮采用**分层锁定**策略：V1 声明中的每条承诺，都在 §7 的工作包里有一个明确的升级路径，且文案结构允许在不改版式的情况下加强措辞。

### 2.2 明确不写的内容（护栏）

以下内容**不得**出现在本轮任何声明文案中：

| 不写 | 原因 |
|---|---|
| 「自动删除会话内容」 | `conversations` / `long_tasks` 无 TTL、无清理任务，归档仅为软删除（`status=2`） |
| 「不与任何第三方共享」 | 知识库正文发往 SiliconFlow，扫描件页面图像发往 MiniMax |
| 「数据不出境 / 国内服务器」 | 后端在海外，用户已选择不披露处理位置 |
| 任何认证徽章 | 无 SOC 2 / ISO 27001 / 等保三级 |
| 「端到端加密」 | 有传输加密与密码 AES-GCM，但不是 E2E |
| 「第三方查询只发检索式」 | 表述为真但会引起「数据未离开服务器」的误读，见 §6 |

### 2.3 竞品校准（依据）

> **证据强度说明**：以下结论来自搜索结果摘要，未逐页打开核实（研究环境无法直接抓取目标页）。文案句式可借鉴，但**不建议直接引用其原话作为对外论据**。

- **PatSnap 智慧芽 Eureka** 的产品页带有「never trained on your data」类表述，同时另设独立 Trust Center 承载认证信息——「产品页一句人话承诺 + 独立合规页承接尽调」的双面结构值得沿用。
- **LexisNexis PatentSight+ / Protégé** 将训练承诺与留存期限写成具体数字（如 90 天留存），是同类中把承诺量化的做法。
- **中文市场**（专利Pro 等信源）给出买家三项评估维度：权属清晰 + 不用于训练 + 认证；并明确驳斥「只有本地私有化部署才安全」。
- **OpenAI API / Anthropic API** 默认均不使用 API 流量训练模型，故「不用于训练」这条承诺在当前架构下**可诚实作出**。但「零留存」不可承诺（上游默认 30 天，ZDR 需单独审批）。

---

## 3. 前端设计（`frontend/nextjs`）

### 3.1 文件清单

| 动作 | 文件 | 说明 |
|---|---|---|
| 增 | `lib/trustNotice.js` | **三条承诺的唯一事实来源** + 两个短句变体，见 §3.2 |
| 增 | `lib/trustNotice.test.mjs` | 回归测试，见 §8.1 |
| 增 | `components/app/TrustNotice.tsx` | 信任模块组件，`'use client'`，从 `lib/trustNotice.js` 取文案 |
| 增 | `app/(landing)/security/page.tsx` | 静态页，`metadata` + `JsonLd`，服务端组件 |
| 改 | `components/app/ChatLanding.tsx` | 在 `chat-landing-slogan` 与 `chat-landing-composer` 之间插入 `<TrustNotice />` |
| 改 | `components/app/LoginForm.tsx` | 提交按钮下方加信任行（`LoginModal` 通过 `.modal-body` 渲染 `LoginForm`，见 `LoginModal.tsx:29`） |
| 改 | `components/app/ChatComposer.tsx` | textarea 下方加聚焦态信任提示 |
| 改 | `components/app/AppLayout.tsx` | 侧边栏 `.sidebar-footer`（`AppLayout.tsx:303`）加「信息安全」入口，见 §3.6 |
| 改 | `lib/app-i18n/locales/zh.ts` | 新增 `security.*` 页键；`security.promises` 从 `lib/trustNotice.js` 取 |
| 改 | `lib/app-i18n/locales/en.ts` | 同上 |
| 改 | `styles/app.css` | `.trust-notice` 系列样式 |

### 3.2 文案的存放位置（有意偏离仓库惯例）

三条承诺**不写在 i18n locale 文件里**，而是放在 `lib/trustNotice.js`，由 `TrustNotice.tsx` 与 `/security` 页共同引用。理由是仓库的测试惯例倒逼的：

- `lib/app-i18n/locales/*.ts` 是 TypeScript。`tsconfig.json:19` 为 `"moduleResolution": "bundler"`，该模式**不允许** import 时带 `.ts` 扩展名；而 `node --test` 解析 ESM 相对导入**必须**带扩展名（实测 Node v22.22.3：`import zh from './locales/zh'` → `ERR_MODULE_NOT_FOUND`；`'./locales/zh.ts'` → 成功）。两者直接冲突。
- 仓库既有的 30+ 个 `lib/*.js` + `lib/*.test.mjs` 全部是纯 JS 模块 + Node 内置测试运行器（`node --test`，无 vitest/jest 配置），零配置摩擦。把文案放进去，既符合惯例，又让 §8.1 的回归测试成为可能。

文案结构：

```js
// lib/trustNotice.js
export const TRUST_COPY = {
  zh: {
    headline: '您的专有信息只留在您的账号里',
    items: [
      { key: 'train',   label: '不外传', desc: '提问与分析结果不用于训练任何 AI 模型' },
      { key: 'isolate', label: '不外泄', desc: '对话记录按账号隔离，仅您本人可读取' },
      { key: 'upload',  label: '不外流', desc: '上传的专利文件处理完成后从服务器删除' },
    ],
    more: '了解更多',
  },
  en: { /* 同结构 */ },
}

export const TRUST_NOTES = {
  zh: {
    login: '🔒 传输全程加密 · 您的内容不会用于训练 AI 模型',
    focus: '您的内容不会用于训练 AI 模型，上传文件处理完成后删除',
  },
  en: { /* 同结构 */ },
}

export const TRUST_VARIANTS = ['card', 'inline', 'hint']
```

**同步约束**：`TRUST_NOTES` 的两句短句是 `items` 的**子集**（受界面空间限制的缩写），不是独立文案。修改 `items` 时必须同步核对短句是否仍然成立——这是本方案唯一的文案同步点，需在模块内以注释标明，并由 §8.1 的测试兜底。

`TrustNotice` 接受一个 `variant` prop：

| variant | 位置 | 形态 |
|---|---|---|
| `card` | 首屏（默认） | `headline` + `items` 三条（移动端竖排 / 桌面端横排）+ `more` 链接 |
| `inline` | 登录弹窗 | 单行小字，取 `TRUST_NOTES[lang].login` |
| `hint` | 输入框聚焦 | 单行小字，取 `TRUST_NOTES[lang].focus` |

`zh` / `en` 的选取沿用 `useI18n()` 提供的 `lang`，不新增语言判断逻辑。

### 3.3 视觉

**克制优先**。首屏是工作界面不是营销页，用力过猛会降低可信度。

- 容器：`teal-50` 浅底 + 左侧 3px teal 竖线，无阴影、无渐变、无圆角重音
- 标题：`teal-700`，正文 15px 半粗
- 三条承诺：`gray-600`，13–14px；每条前缀一个 stroke-2px 线性图标，风格与 `ChatLanding.tsx` 现有 `ICONS` 一致
- 竖排（移动端 `<768px`）/ 横排（桌面端）由 CSS media query 切换，不用 JS 断点
- 「了解更多 →」右对齐小字，链到 `/security`

### 3.4 登录弹窗触点

`LoginModal.tsx` 内 `LoginForm` 组件下方，主提交按钮之后：

```
[ 登录 ]
🔒 传输全程加密 · 您的内容不会用于训练 AI 模型
```

居中小字，`gray-500`，12px。**不做链接**——弹窗内跳转会让用户丢失表单状态。

### 3.5 输入框聚焦触点

`ChatComposer.tsx` 的 textarea 下方：

- 绑定 `onFocus` → `setShowTrustHint(true)`，`onBlur` **不清除**（清洗：避免反复弹扰），组件卸载即丢失
- 淡入动画 `opacity 0 → 1`，`transition: opacity 150ms`；`prefers-reduced-motion: reduce` 时禁用动画
- 移动端不展示（输入区空间紧张，且键盘弹起后提示会被遮挡）
- 已有输入内容时不展示（用户已在专心打字，不再需要说服）

### 3.6 `/security` 的可达性（必做）

**问题**：应用内目前**没有任何**指向 `/security` 的入口。`AppLayout` 是侧边栏布局，只使用 `.sidebar-footer` 放「开发者模式」开关（`AppLayout.tsx:303`），没有站点页脚。营销页 `LandingPage.tsx` 的页脚虽然有隐私政策链接，但首页走的是应用而非营销页。若不处理，`/security` 将成为孤岛——只有从信任模块点「了解更多」才能到达，而侧边栏里的老用户永远看不到。

**处理**：在 `AppLayout` 的 `.sidebar-footer` 内、「开发者模式」开关下方，加一个 `nav-item` 样式的「信息安全」链接，指向 `/security`。复用现有 `nav-item` 类，不加新样式。

**侧边栏折叠态**：折叠时只显示图标（盾牌线性图标，与 `ChatLanding.tsx` 的 `ICONS` 风格一致），`title` 属性提供完整文本。

---

## 4. `/security` 页结构

服务端组件，纯静态，双语。路由 `app/(landing)/security/page.tsx`。复用 `(landing)/layout.tsx` 的 `LandingI18nProvider`。

1. **Hero** — 主标语 + 一句话承诺
2. **三条承诺的展开**（每条：一句人话 + 一句技术说明）

   | 承诺 | 展开文案 |
   |---|---|
   | 不外传 | 「您的提问、上传文件与分析结果，不会被用于训练任何 AI 模型。模型推理通过企业级 API 通道完成，该通道默认不保留训练用途。」 |
   | 不外泄 | 「对话记录按账号严格隔离，每次读取均按账号过滤。上传文件仅您本人可访问。」 |
   | 不外流 | 「上传的专利文件在分析处理完成后从服务器删除。我们不会将其用于任何其他用途。」 |

3. **传输与存储** — 全部请求经 HTTPS 加密传输；登录密码以 AES-GCM 加密后传输，服务端不落明文
4. **日志** — 诊断日志中的提问内容截断保留前 80 字符。（此条为主动披露，是可信度加分项）
5. **版本与更新日期** — `Version 1.0 · Last updated 2026-09-16`，加一行「我们会持续加强数据保护措施，本页随能力升级更新」

### 4.1 用户未拍板项的处理

§6 列出的「第三方查询只发检索式」在 V1 **不出现**。留待 W2 完成后以改写版本补入。

---

## 5. `/privacy-policy` 重写

### 5.1 现状问题

`app/(landing)/privacy-policy/page.tsx` 为**旧产品线文案**：

- 通篇描述 Chrome 插件的数据采集（`page.tsx:58` "how the CopiioAI Chrome extension collects…"）
- 定位为开发者工具（`page.tsx:53` "an AI-powered developer tool designed to help users turn APIs and web requests into reusable AI tools"）
- 纯英文，`Last updated: 2026-03-05`
- `metadata.keywords` 含 `'Chrome extension privacy'`

与当前「AI 专利检索与分析」定位完全不符。客户尽调时点入会质疑产品线一致性。

### 5.2 本轮处理

- 重写正文为专利数据口径（提问、上传文件、分析结果、知识库四类数据）
- 双语
- 更新 `metadata` / `openGraph` / `keywords`
- 更新 `Last updated`
- 保留既有 9 节结构骨架，替换内容与措辞

**边界**：重写不等于放宽。凡 §2.2 护栏禁止的内容，同样不得写入隐私政策。隐私政策与 `/security` 页的口径必须一致——两处说法打架比少写更伤。

---

## 6. 关于「第三方查询只发检索式」

**事实**：用户提问经 LLM 改写为结构化检索式后发送至外部专利数据库（`search_query_builder.py:333`、`react_tools.py:1070`），外部数据源确实不接收用户原话。

**但**：改写动作本身即把原话发往 LLM（`search_query_builder.py:347`）。该表述虽逐字为真，却会让读者推出「提问未离开服务器」的错误结论。被追问「你们的 AI 是本地跑的吗」时反而更难解释。

**处理**：V1 不出现。W2 完成后改写为：

> 您的提问仅用于理解检索意图；向外部专利数据库发起的查询为结构化检索条件。提问内容不会被写入检索日志。

保留「原话没有被到处乱发」这一真实优点，同时不暗示数据未出服务器。

---

## 7. 后续工作包（分层锁定）

| 工作包 | 内容 | 解锁的新承诺 |
|---|---|---|
| **W1（本次）** | 文案 + 三处触点 + `/security` + privacy-policy 重写 | 不外传 / 不外泄 / 不外流 |
| **W2** | ① `api_routes/core.py:728,1260,978` 日志脱敏 + 日志轮转；② `conversations` / `long_tasks` 真实过期删除；③ 上传目录兜底 GC（覆盖崩溃/超时路径，`celery_worker.py:1349` 之外） | 升级为「会话到期自动删除」；「不外流」由「处理完成后删除」升级为「**保证**删除」；补回 §6 改写条目 |
| **W3** | 审计日志；知识库默认不公开（现 `public=1` 默认他人可读） | 「操作可审计」；「知识库默认私有」 |
| **W4（远期）** | 等保三级 / ISO 27001 / SOC 2 | 展示认证徽章 |

### 7.1 与声明的耦合点

W2 完成后需同步修改：`lib/trustNotice.js` 的 `TRUST_COPY`、`/security` 页对应段落、`Version` 号。文案结构已为此预留（三条承诺各自独立成条，升级时只改措辞不改版式）。`TRUST_NOTES` 的短句若涉及被升级的条目，须一并核对（见 §3.2 同步约束）。

---

## 8. 验证方式

| 验收项 | 方式 |
|---|---|
| 三个触点均展示、文案一致 | 本地 `npm run dev`，首屏 / 登录弹窗 / 输入框聚焦三处目视 |
| 文案无护栏违规 | 对照 §2.2 逐条检查中英文两版 |
| 中英文切换正确 | 语言切换后三处同步更新 |
| `/security` 可达且 SSR 可爬 | 构造后 `curl` 检查纯 HTML 含正文；应用侧边栏入口（§3.6）可点击到达 |
| 移动端不挤压输入区 | 375px 宽度下首屏检查；输入框聚焦提示在移动端不出现 |
| 无动画偏好生效 | 系统开启 reduced-motion 后淡入动画禁用 |
| 构建通过 | `npm run build`（`frontend/nextjs`） |
| 单元测试通过 | `node --test lib/trustNotice.test.mjs` |

### 8.1 回归测试

`lib/trustNotice.test.mjs`，用 Node 内置运行器（`node --test lib/trustNotice.test.mjs`，与仓库既有 `lib/*.test.mjs` 同一套，无 vitest/jest 依赖）。覆盖：

1. `TRUST_COPY.zh` / `TRUST_COPY.en` 均存在，且 `headline`、`more` 非空字符串
2. 两种语言的 `items` **长度相同**，且各条 `key` **集合一致**（防止加了一条承诺只改了中文）
3. 每条 `items` 的 `key` / `label` / `desc` 均为非空字符串
4. `TRUST_NOTES.zh` / `TRUST_NOTES.en` 的 `login` 与 `focus` 非空
5. `TRUST_VARIANTS` 恰为 `['card', 'inline', 'hint']`，且 `TrustNotice` 的 `variant` prop 取值不超出该集合

第 2 条是关键——它把 §3.2 的「同步约束」从注释约定变成会失败的测试。
