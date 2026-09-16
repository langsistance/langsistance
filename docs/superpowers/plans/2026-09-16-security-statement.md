# 信息安全声明 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在用户即将输入专有技术方案的位置，给出三条可被代码验证的保密承诺，并提供承载细节的 `/security` 页。

**Architecture:** 三条承诺的文案抽到纯 JS 模块 `lib/trustNotice.js`（唯一事实来源），由一个 `TrustNotice` 组件以三种 variant 渲染到首屏 / 登录表单 / 输入框聚焦三处；`/security` 页与隐私政策页承载详细条款。文案只写当前后端能力能支撑的内容，其余留待 W2/W3 工作包。

**Tech Stack:** Next.js 15 App Router（RSC + `'use client'`）、React 18、TypeScript、原生 CSS（`styles/app.css`，第 1 行 `@import './popup.css'`）、Node 内置测试运行器（`node --test`，无 vitest/jest）。

## Global Constraints

- 分支：`feat/security-statement`（自 `main` @ `4db28ef` 切出）
- 工作目录：`frontend/nextjs`
- 测试运行器：`node --test`。仓库无 vitest/jest 配置，不要新增测试框架依赖
- **文案护栏（违反即整个方案作废）** —— 以下内容不得出现在任何声明 / 隐私政策文案中：
  - 「自动删除会话内容」
  - 「不与任何第三方共享」
  - 「数据不出境」/「国内服务器」
  - 任何认证徽章（SOC 2 / ISO 27001 / 等保三级）
  - 「端到端加密」
  - 「第三方查询只发检索式」
- **供应商名禁入可见面**：佰腾 / Baiten 及任何上游数据源厂名不得出现在页面文案、metadata、注释中的用户可见位置
- 文案存放：三条承诺只存在于 `lib/trustNotice.js`，不得复制进 `lib/app-i18n/locales/*.ts`
- 组件文件用 `PascalCase.tsx`，纯逻辑模块用 `camelCase.js` + `camelCase.test.mjs`（仓库既有惯例）
- 提交信息用中文，格式 `type(scope): 描述`
- 每个 Task 结束前必须能通过：`node --test lib/trustNotice.test.mjs`（Task 1 之后）

---

### Task 1: 承诺文案模块（唯一事实来源）

**Files:**
- Create: `frontend/nextjs/lib/trustNotice.js`
- Test: `frontend/nextjs/lib/trustNotice.test.mjs`

**Interfaces:**
- Consumes: 无（本仓库第一个任务）
- Produces:
  - `TRUST_COPY: { zh: TrustLocale, en: TrustLocale }`，其中 `TrustLocale = { headline: string, items: TrustItem[], more: string }`，`TrustItem = { key: string, label: string, desc: string }`
  - `TRUST_NOTES: { zh: { login: string, focus: string }, en: { login: string, focus: string } }`

  （variant 名 `'card' | 'inline' | 'hint'` 由 `TrustNotice.tsx` 自己拥有，不放进本模块——没有生产代码消费一个字符串数组，这是 YAGNI。）

- [ ] **Step 1: 写失败的测试**

创建 `frontend/nextjs/lib/trustNotice.test.mjs`：

```js
import test from 'node:test'
import assert from 'node:assert/strict'

import { TRUST_COPY, TRUST_NOTES } from './trustNotice.js'

const LANGS = ['zh', 'en']

test('两种语言都提供 headline 与 more', () => {
  for (const lang of LANGS) {
    assert.equal(typeof TRUST_COPY[lang].headline, 'string', `${lang} headline`)
    assert.ok(TRUST_COPY[lang].headline.trim().length > 0, `${lang} headline 非空`)
    assert.ok(TRUST_COPY[lang].more.trim().length > 0, `${lang} more 非空`)
  }
})

test('两种语言的承诺条数与 key 集合完全一致', () => {
  const keysOf = (lang) => TRUST_COPY[lang].items.map((i) => i.key).sort()
  assert.deepEqual(keysOf('zh'), keysOf('en'))
  assert.equal(TRUST_COPY.zh.items.length, 3)
})

test('每条承诺的 key / label / desc 均为非空字符串', () => {
  for (const lang of LANGS) {
    for (const item of TRUST_COPY[lang].items) {
      assert.ok(item.key.trim().length > 0, `${lang} key`)
      assert.ok(item.label.trim().length > 0, `${lang} ${item.key} label`)
      assert.ok(item.desc.trim().length > 0, `${lang} ${item.key} desc`)
    }
  }
})

test('两种语言都提供 login 与 focus 短句', () => {
  for (const lang of LANGS) {
    assert.ok(TRUST_NOTES[lang].login.trim().length > 0, `${lang} login`)
    assert.ok(TRUST_NOTES[lang].focus.trim().length > 0, `${lang} focus`)
  }
})
```

- [ ] **Step 2: 运行测试确认失败**

Run（在 `frontend/nextjs` 下）：`node --test lib/trustNotice.test.mjs`
Expected: FAIL — `Cannot find module` 或 `ERR_MODULE_NOT_FOUND`，因为 `lib/trustNotice.js` 还不存在

- [ ] **Step 3: 实现模块**

创建 `frontend/nextjs/lib/trustNotice.js`：

```js
// 信息安全声明的文案唯一事实来源。
//
// 为什么不放进 lib/app-i18n/locales/*.ts：那两处是 TypeScript，tsconfig 的
// "moduleResolution": "bundler" 不允许 import 时带 .ts 扩展名，而 node --test
// 解析 ESM 相对导入必须带扩展名，两者冲突。纯 .js 模块同时满足 Next 打包与
// 内置测试运行器。
//
// 同步约束：TRUST_NOTES 是 TRUST_COPY.items 在界面空间受限处的缩写子集，
// 不是独立文案。改动 items 时必须回头核对 TRUST_NOTES 两句是否仍然成立。
// lib/trustNotice.test.mjs 会校验结构一致，但校验不了语义漂移——靠人看。

export const TRUST_COPY = {
  zh: {
    headline: '您的专有信息只留在您的账号里',
    items: [
      { key: 'train', label: '不外传', desc: '提问与分析结果不用于训练任何 AI 模型' },
      { key: 'isolate', label: '不外泄', desc: '对话记录按账号隔离，仅您本人可读取' },
      { key: 'upload', label: '不外流', desc: '上传的专利文件处理完成后从服务器删除' },
    ],
    more: '了解更多',
  },
  en: {
    headline: 'Your proprietary information stays in your account',
    items: [
      { key: 'train', label: 'Not shared', desc: 'Prompts and results are never used to train AI models' },
      { key: 'isolate', label: 'Not exposed', desc: 'Conversations are isolated by account and readable only by you' },
      { key: 'upload', label: 'Not retained', desc: 'Uploaded patent files are deleted from our servers after processing' },
    ],
    more: 'Learn more',
  },
}

export const TRUST_NOTES = {
  zh: {
    login: '🔒 传输全程加密 · 您的内容不会用于训练 AI 模型',
    focus: '您的内容不会用于训练 AI 模型，上传文件处理完成后删除',
  },
  en: {
    login: '🔒 Encrypted in transit · Your content is never used to train AI models',
    focus: 'Your content is never used to train AI models; uploads are deleted after processing',
  },
}
```

- [ ] **Step 4: 运行测试确认通过**

Run（在 `frontend/nextjs` 下）：`node --test lib/trustNotice.test.mjs`
Expected: PASS — `# pass 4` / `# fail 0`

- [ ] **Step 5: 提交**

```bash
git add frontend/nextjs/lib/trustNotice.js frontend/nextjs/lib/trustNotice.test.mjs
git commit -m "feat(security): 保密承诺文案模块 —— 唯一事实来源 + 结构回归测试"
```

---

### Task 2: TrustNotice 组件 + 首屏接入

**Files:**
- Create: `frontend/nextjs/components/app/TrustNotice.tsx`
- Modify: `frontend/nextjs/components/app/ChatLanding.tsx`（第 68-71 行之间插入）
- Modify: `frontend/nextjs/styles/app.css`（文件末尾追加）

**Interfaces:**
- Consumes: `TRUST_COPY`、`TRUST_NOTES`（Task 1）；`useI18n()`（既有，返回 `{ lang, t }`）
- Produces: `TrustNotice` 默认导出，props 为 `{ variant?: 'card' | 'inline' | 'hint' }`；`variant` 缺省为 `'card'`。三种 variant 的 DOM 结构见 Step 3，Task 3/4 只使用 `variant="inline"` 与 `variant="hint"`

- [ ] **Step 1: 写组件**

创建 `frontend/nextjs/components/app/TrustNotice.tsx`：

```tsx
'use client'

import { useI18n } from '@/lib/app-i18n'
import { TRUST_COPY, TRUST_NOTES } from '@/lib/trustNotice'

// 线性图标，风格对齐 ChatLanding.tsx 的 ICONS（stroke 2px，24 视图框）
const ICONS: Record<string, JSX.Element> = {
  train: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="11" width="18" height="11" rx="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  ),
  isolate: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  ),
  upload: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 6h18" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  ),
}

// 用字面量联合而非从 .js 模块 import 类型：trustNotice.js 是纯 JS，
// 类型靠 JSDoc，跨文件 typeof import 容易在 next build 下解析失败。
interface TrustNoticeProps {
  variant?: 'card' | 'inline' | 'hint'
}

export default function TrustNotice({ variant = 'card' }: TrustNoticeProps) {
  const { lang } = useI18n()
  const locale = lang === 'en' ? 'en' : 'zh'

  if (variant === 'inline') {
    return <p className="trust-inline">{TRUST_NOTES[locale].login}</p>
  }

  if (variant === 'hint') {
    return <p className="trust-hint">{TRUST_NOTES[locale].focus}</p>
  }

  const copy = TRUST_COPY[locale]

  return (
    <section className="trust-notice" aria-label={copy.headline}>
      <h3 className="trust-notice-headline">{copy.headline}</h3>
      <ul className="trust-notice-items">
        {copy.items.map((item) => (
          <li key={item.key} className="trust-notice-item">
            <span className="trust-notice-icon">{ICONS[item.key]}</span>
            <span className="trust-notice-label">{item.label}</span>
            <span className="trust-notice-desc">{item.desc}</span>
          </li>
        ))}
      </ul>
      <a className="trust-notice-more" href="/security">{copy.more} →</a>
    </section>
  )
}
```

`lib/trustNotice.js`（Task 1 产物）无需改动。`TrustNotice.tsx` 不从中 import 类型——`variant` 的字面量联合在组件内就地声明，理由见组件内注释。

- [ ] **Step 2: 追加样式**

在 `frontend/nextjs/styles/app.css` **文件末尾**追加：

```css
/* ── 信息安全声明 ─────────────────────────────────────────────
   三处触点共用：card（首屏）/ inline（登录表单）/ hint（输入框聚焦）
   克制的视觉：不加阴影、不加渐变，避免读起来像营销位 */

.trust-notice {
  max-width: 680px;
  margin: 0 auto 28px;
  padding: 16px 20px;
  background: #f0fdfa;
  border-left: 3px solid #0d9488;
  text-align: left;
}

.trust-notice-headline {
  margin: 0 0 12px;
  font-size: 15px;
  font-weight: 600;
  color: #0f766e;
}

.trust-notice-items {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.trust-notice-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #4b5563;
}

.trust-notice-icon {
  display: inline-flex;
  color: #0d9488;
  flex-shrink: 0;
}

.trust-notice-label {
  font-weight: 600;
  color: #374151;
  flex-shrink: 0;
}

.trust-notice-more {
  display: inline-block;
  margin-top: 12px;
  font-size: 12px;
  color: #0d9488;
  text-decoration: none;
}

.trust-notice-more:hover {
  text-decoration: underline;
}

.trust-inline {
  margin: 16px 0 0;
  text-align: center;
  font-size: 12px;
  color: #6b7280;
  line-height: 1.5;
}

.trust-hint {
  margin: 6px 2px 0;
  font-size: 12px;
  color: #9ca3af;
  animation: trust-hint-fade 150ms ease-out;
}

@keyframes trust-hint-fade {
  from { opacity: 0; }
  to   { opacity: 1; }
}

@media (prefers-reduced-motion: reduce) {
  .trust-hint { animation: none; }
}

@media (max-width: 767px) {
  .trust-notice {
    padding: 14px 16px;
    margin-bottom: 20px;
  }
  .trust-notice-item {
    align-items: flex-start;
  }
}
```

- [ ] **Step 3: 接入首屏**

在 `frontend/nextjs/components/app/ChatLanding.tsx` 中：

1. 加 import（放在 `import PatentOnboardingWizard from './PatentOnboardingWizard'` 之后）：

```tsx
import TrustNotice from './TrustNotice'
```

2. 在 slogan 与 composer 之间插入（即当前第 68 行 `</h2>` 之后、第 69 行 `<div className="chat-landing-composer">` 之前）：

```tsx
      <TrustNotice />
```

改完后该段应为：

```tsx
    <div className="chat-landing">
      <h2 className="chat-landing-slogan">{t('chat.landing.slogan')}</h2>
      <TrustNotice />
      <div className="chat-landing-composer">
        <ChatComposer {...composerProps} />
      </div>
```

- [ ] **Step 4: 目视验证**

Run（在 `frontend/nextjs` 下）：`npm run dev`
在浏览器打开 `http://localhost:3000`，确认：

1. slogan「专利情报，一问即得」下方出现信任卡片
2. 三条承诺各带图标，文案与 `lib/trustNotice.js` 一致
3. 点「了解更多 →」跳转到 `/security`（此时该页还是 404，属预期，Task 5 补齐）
4. 切到英文（右上角语言按钮），文案变英文
5. 浏览器窗口缩到 375px 宽，三条承诺竖排且不溢出

Expected: 上述 5 项全部符合

- [ ] **Step 5: 提交**

```bash
git add frontend/nextjs/components/app/TrustNotice.tsx frontend/nextjs/components/app/ChatLanding.tsx frontend/nextjs/lib/trustNotice.js frontend/nextjs/styles/app.css
git commit -m "feat(security): 信任模块组件并接入首屏 —— 标语与输入框之间"
```

---

### Task 3: 登录表单触点

**Files:**
- Modify: `frontend/nextjs/components/app/LoginForm.tsx`（第 190-201 行的 `.login-footer` 之后）

**Interfaces:**
- Consumes: `TrustNotice` with `variant="inline"`（Task 2）
- Produces: 无下游依赖

- [ ] **Step 1: 接入**

在 `frontend/nextjs/components/app/LoginForm.tsx` 中：

1. 加 import（放在 `import LanguageToggleButton from '@/components/app/LanguageToggleButton'` 之后）：

```tsx
import TrustNotice from '@/components/app/TrustNotice'
```

2. 在 `.login-footer` 那个 `<p>` 之后、`</div>`（`.login-card` 收尾）之前插入。即当前第 201 行 `</p>` 之后：

```tsx
        <TrustNotice variant="inline" />
```

改完后该段应为：

```tsx
        <p className="login-footer">
          {isSignUp
            ? (lang === 'en' ? 'Already have an account?' : '已有账号？')
            : (lang === 'en' ? "Don't have an account?" : '没有账号？')}
          <button onClick={() => {
            setIsSignUp(!isSignUp)
            setConfirmPassword('')
            setError('')
          }}>
            {isSignUp ? (lang === 'en' ? 'Sign In' : '登录') : (lang === 'en' ? 'Sign Up' : '注册')}
          </button>
        </p>

        <TrustNotice variant="inline" />
      </div>
    </div>
  )
}
```

- [ ] **Step 2: 目视验证**

Run（在 `frontend/nextjs` 下）：`npm run dev`
触发登录弹窗的方式：在首屏输入框随便输入文字后按回车，会弹出登录框。
确认：

1. 「没有账号？注册」下方出现一行「🔒 传输全程加密 · 您的内容不会用于训练 AI 模型」
2. 点击「注册」切换到注册态，该行文字位置不变、依然在底部
3. 切英文后该行变英文

Expected: 上述 3 项全部符合

- [ ] **Step 3: 提交**

```bash
git add frontend/nextjs/components/app/LoginForm.tsx
git commit -m "feat(security): 登录表单补信任行 —— 交出账号密码前的最后一句"
```

---

### Task 4: 输入框聚焦触点

**Files:**
- Modify: `frontend/nextjs/components/app/ChatComposer.tsx`（第 141-150 行的 textarea 与其后的发送按钮之间）

**Interfaces:**
- Consumes: `TrustNotice` with `variant="hint"`（Task 2）
- Produces: 无下游依赖

- [ ] **Step 1: 接入**

在 `frontend/nextjs/components/app/ChatComposer.tsx` 中：

1. 把第 3 行改为（增加 `useState`）：

```tsx
import { useEffect, useRef, useState } from 'react'
```

2. 加 import（放在 `import { useI18n } from '@/lib/app-i18n'` 之后）：

```tsx
import TrustNotice from './TrustNotice'
```

3. 在 `const fileInputRef = useRef<HTMLInputElement | null>(null)`（第 31 行）之后加：

```tsx
  // 聚焦后不因失焦隐藏：反复弹扰比不提示更糟。组件卸载即重置。
  const [showTrustHint, setShowTrustHint] = useState(false)
```

4. 给 textarea 加 `onFocus`。当前第 141-150 行的 textarea 改为：

```tsx
        <textarea
          ref={textareaRef}
          className="chat-input"
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          onPaste={handleFilePaste}
          onFocus={() => setShowTrustHint(true)}
          placeholder={t('chat.placeholder')}
          rows={1}
        />
```

5. 在 `</div>`（`.chat-input-wrapper` 收尾，当前第 173 行）之后、`</>` 之前插入提示：

```tsx
      </div>
      {showTrustHint && !input && <TrustNotice variant="hint" />}
    </>
  )
}
```

- [ ] **Step 2: 补移动端隐藏规则**

`variant="hint"` 在移动端不展示（键盘弹起后会遮挡提示，且输入区空间紧张）。在 `frontend/nextjs/styles/app.css` 中 `.trust-hint` 规则**之后**追加：

```css
/* 移动端不展示聚焦提示：键盘弹起后会被遮挡，且输入区空间紧张 */
@media (max-width: 767px) {
  .trust-hint { display: none; }
}
```

- [ ] **Step 3: 目视验证**

Run（在 `frontend/nextjs` 下）：`npm run dev`
确认：

1. 页面加载后**不显示**提示（未聚焦）
2. 点击输入框 → 出现「您的内容不会用于训练 AI 模型，上传文件处理完成后删除」
3. 点页面别处使输入框失焦 → 提示**仍然可见**（不隐藏）
4. 在输入框输入任意字符 → 提示消失
5. 清空输入框 → 提示**不**重新出现（已聚焦过就不重复提示）
6. DevTools 切到 375px 宽并刷新 → 聚焦输入框，提示不出现

Expected: 上述 6 项全部符合

- [ ] **Step 4: 提交**

```bash
git add frontend/nextjs/components/app/ChatComposer.tsx frontend/nextjs/styles/app.css
git commit -m "feat(security): 输入框聚焦信任提示 —— 敲进技术方案前的那一刻"
```

---

### Task 5: `/security` 信息安全页

**Files:**
- Create: `frontend/nextjs/app/(landing)/security/page.tsx`

**Interfaces:**
- Consumes: `TRUST_COPY`（Task 1）；`JsonLd`（既有，`components/JsonLd.tsx`，用法见 `app/(landing)/privacy-policy/page.tsx:25-45`）；`(landing)/layout.tsx` 已提供 `LandingI18nProvider`
- Produces: 路由 `/security`

- [ ] **Step 1: 写页面**

创建 `frontend/nextjs/app/(landing)/security/page.tsx`：

```tsx
import type { Metadata } from 'next'
import JsonLd from '@/components/JsonLd'

const LAST_UPDATED = '2026-09-16'
const VERSION = '1.0'

export const metadata: Metadata = {
  title: 'Information Security',
  description:
    'How CopiioAI protects your patent data — no AI training on your content, account-isolated conversations, and uploads deleted after processing.',
  keywords: [
    'patent data security', 'AI patent confidentiality', 'CopiioAI security',
    'patent search privacy', 'IP data protection',
  ],
  openGraph: {
    title: 'Information Security | CopiioAI',
    description:
      'No AI training on your content, account-isolated conversations, uploads deleted after processing.',
    url: 'https://copiioai.com/security',
    siteName: 'CopiioAI',
    type: 'website',
  },
  alternates: {
    canonical: 'https://copiioai.com/security',
  },
}

const COPY = {
  zh: {
    langLabel: '中文',
    h1: '信息安全',
    intro: '专利是您最重要的资产之一。以下是我们对您数据的承诺，以及每条承诺对应的具体做法。',
    promises: [
      {
        title: '不外传 —— 不用于训练任何 AI 模型',
        body: '您的提问、上传文件与分析结果，不会被用于训练任何 AI 模型。模型推理通过企业级 API 通道完成，该通道默认不保留数据用于训练用途。',
      },
      {
        title: '不外泄 —— 会话按账号隔离',
        body: '对话记录按账号严格隔离，每次读取均按账号过滤。上传文件仅您本人可访问。',
      },
      {
        title: '不外流 —— 上传文件处理后删除',
        body: '上传的专利文件在分析处理完成后从服务器删除。我们不会将其用于任何其他用途。',
      },
    ],
    transportTitle: '传输与存储',
    transportItems: [
      '所有请求经 HTTPS 加密传输。',
      '登录密码以 AES-GCM 加密后传输，服务端不落明文。',
    ],
    logsTitle: '日志',
    logsBody: '用于诊断的日志中，提问内容仅保留前 80 个字符。',
    footerNote: '我们会持续加强数据保护措施，本页随能力升级更新。',
    versionLine: `Version ${VERSION} · Last updated ${LAST_UPDATED}`,
  },
  en: {
    langLabel: 'English',
    h1: 'Information Security',
    intro: 'Patents are among your most valuable assets. Here is what we commit to — and how each commitment is implemented.',
    promises: [
      {
        title: 'Not shared — never used to train AI models',
        body: 'Your prompts, uploaded files, and analysis results are never used to train any AI model. Model inference runs through enterprise API channels, which by default do not retain data for training.',
      },
      {
        title: 'Not exposed — conversations isolated by account',
        body: 'Conversation records are strictly isolated by account, and every read is filtered by account. Uploaded files are accessible only to you.',
      },
      {
        title: 'Not retained — uploads deleted after processing',
        body: 'Uploaded patent files are deleted from our servers once analysis finishes. We do not use them for any other purpose.',
      },
    ],
    transportTitle: 'Transport and storage',
    transportItems: [
      'All requests are transmitted over HTTPS.',
      'Login passwords are encrypted with AES-GCM in transit; no plaintext is stored server-side.',
    ],
    logsTitle: 'Logging',
    logsBody: 'In diagnostic logs, prompt content is truncated to the first 80 characters.',
    footerNote: 'We continue to strengthen our data protection measures. This page is updated as capabilities improve.',
    versionLine: `Version ${VERSION} · Last updated ${LAST_UPDATED}`,
  },
} as const

type CopyLang = keyof typeof COPY

function SecurityCopy({ copy }: { copy: (typeof COPY)[CopyLang] }) {
  return (
    <section className="mb-14">
      <h2 className="text-2xl font-bold text-gray-900 mb-4">{copy.h1}</h2>
      <p className="text-gray-700 mb-8">{copy.intro}</p>

      <div className="space-y-6">
        {copy.promises.map((p) => (
          <div key={p.title} className="border-l-2 border-teal-600 pl-5">
            <h3 className="text-lg font-semibold text-teal-700 mb-2">{p.title}</h3>
            <p className="text-gray-700">{p.body}</p>
          </div>
        ))}
      </div>

      <h3 className="text-xl font-bold text-gray-900 mt-10 mb-3">{copy.transportTitle}</h3>
      <ul className="list-disc pl-6 text-gray-700 space-y-1">
        {copy.transportItems.map((t) => (
          <li key={t}>{t}</li>
        ))}
      </ul>

      <h3 className="text-xl font-bold text-gray-900 mt-10 mb-3">{copy.logsTitle}</h3>
      <p className="text-gray-700">{copy.logsBody}</p>

      <p className="text-gray-500 text-sm mt-10">{copy.footerNote}</p>
      <p className="text-gray-400 text-xs mt-2">{copy.versionLine}</p>
    </section>
  )
}

export default function SecurityPage() {
  return (
    <>
      <JsonLd
        id="jsonld-security"
        data={{
          '@context': 'https://schema.org',
          '@type': 'WebPage',
          name: 'Information Security',
          description:
            'How CopiioAI protects your patent data — no AI training on your content, account-isolated conversations, and uploads deleted after processing.',
          publisher: {
            '@type': 'Organization',
            name: 'CopiioAI',
            url: 'https://copiioai.com',
          },
        }}
      />
      <div className="max-w-3xl mx-auto px-6 py-16">
        <nav className="mb-5 text-sm text-gray-500">
          <a href="/" className="text-teal-600 hover:underline">CopiioAI</a>
          {' / '}
          <span>Information Security</span>
        </nav>

        <h1 className="text-3xl font-bold text-gray-900 mb-10">Information Security</h1>

        <SecurityCopy copy={COPY.zh} />
        <div className="border-t border-gray-200 pt-4" />
        <SecurityCopy copy={COPY.en} />

        <div className="mt-12 text-sm">
          <a href="/privacy-policy" className="text-teal-600 hover:underline">Privacy Policy</a>
        </div>
      </div>
    </>
  )
}
```

**说明**：本页中英双语上下并列渲染。原因有二——该路由在 `(landing)` 分支下未接入 `app-i18n` 的 `useI18n`（只有 `LandingI18nProvider`，其语言状态作用于营销页组件），且合规类页面同时呈现两种语言是常见做法，无需用户切换即可读到。

- [ ] **Step 2: 目视验证**

Run（在 `frontend/nextjs` 下）：`npm run dev`
打开 `http://localhost:3000/security`，确认：

1. 页面可达（不是 404）
2. 中文段落在上、英文段落在下，各有标题
3. 三条承诺各带 teal 左边线
4. 底部显示 `Version 1.0 · Last updated 2026-09-16`
5. 「Privacy Policy」链接可达
6. 从首屏点「了解更多 →」能到达本页

Expected: 上述 6 项全部符合

- [ ] **Step 3: 验证 SSR 可爬（爬虫不执行 JS）**

Run（在 `frontend/nextjs` 下）：

```bash
npm run build && npx next start -p 3100 &
sleep 5
curl -s http://localhost:3100/security | grep -c "不外传"
kill %1
```

Expected: 输出 ≥ `1`（说明纯 HTML 里含正文，百度/AI 搜索不执行 JS 也能读到）

- [ ] **Step 4: 提交**

```bash
git add "frontend/nextjs/app/(landing)/security/page.tsx"
git commit -m "feat(security): 新增 /security 信息安全页 —— 中英双语条款 + 版本日期"
```

---

### Task 6: 侧边栏入口（避免页面成为孤岛）

**Files:**
- Modify: `frontend/nextjs/components/app/AppLayout.tsx`（第 303-317 行的 `.sidebar-footer`）

**Interfaces:**
- Consumes: 路由 `/security`（Task 5）
- Produces: 无下游依赖

- [ ] **Step 1: 接入**

在 `frontend/nextjs/components/app/AppLayout.tsx` 的 `.sidebar-footer` 中，`</button>`（开发者模式按钮收尾，当前第 316 行）之后插入：

```tsx
            <a
              className="nav-item"
              href="/security"
              title={lang === 'en' ? 'Information Security' : '信息安全'}
            >
              <span>{lang === 'en' ? 'Information Security' : '信息安全'}</span>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            </a>
```

改完后该段应为：

```tsx
          <div className="sidebar-footer">
            <button
              className="nav-item"
              onClick={toggleDevMode}
              style={{ cursor: 'pointer' }}
              title={t('developer.pattern')}
            >
              <span>{t('developer.pattern')}</span>
              <div className="switch-wrap">
                <div className={`switch-container${devMode ? ' active' : ''}`}>
                  <div className="switch-slider" />
                </div>
              </div>
            </button>
            <a
              className="nav-item"
              href="/security"
              title={lang === 'en' ? 'Information Security' : '信息安全'}
            >
              <span>{lang === 'en' ? 'Information Security' : '信息安全'}</span>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            </a>
          </div>
```

**注意**：`lang` 需在 `AppLayout` 作用域内可用。先确认该组件第 296 行附近是否已有 `const { t, lang } = useI18n()`；若只有 `t`，改为 `const { t, lang } = useI18n()`。

- [ ] **Step 2: 目视验证**

Run（在 `frontend/nextjs` 下）：`npm run dev`
打开 `http://localhost:3000`，登录后确认：

1. 左侧边栏底部「开发者模式」开关下方出现「信息安全」条目
2. 点击跳转到 `/security`
3. 折叠侧边栏 → 只显示盾牌图标，悬停显示完整标题

   （无需新增 CSS：`styles/popup.css:421` 的 `.sidebar.sidebar-collapsed .nav-item span { display: none }` 已覆盖本条目——它用的正是 `<span>` + svg 结构。）

4. 切换语言 → 条目文字在中英之间切换

Expected: 上述 4 项全部符合

- [ ] **Step 3: 提交**

```bash
git add frontend/nextjs/components/app/AppLayout.tsx
git commit -m "feat(security): 侧边栏加信息安全入口 —— 此前 /security 无任何应用内入口"
```

---

### Task 7: 隐私政策重写

**Files:**
- Modify: `frontend/nextjs/app/(landing)/privacy-policy/page.tsx`（整份替换）

**Interfaces:**
- Consumes: 无
- Produces: 无下游依赖

**背景**：现文件是**旧产品线文案**——通篇讲 Chrome 插件采集什么数据（`page.tsx:58`）、定位为「开发者工具」（`page.tsx:53`），`keywords` 含 `'Chrome extension privacy'`。与当前「AI 专利检索与分析」定位不符，尽调时会引发产品线一致性质疑。

- [ ] **Step 1: 整份替换**

用以下内容替换 `frontend/nextjs/app/(landing)/privacy-policy/page.tsx` 全文：

```tsx
import type { Metadata } from 'next'
import JsonLd from '@/components/JsonLd'

const LAST_UPDATED = '2026-09-16'

export const metadata: Metadata = {
  title: 'Privacy Policy',
  description:
    'Privacy Policy for CopiioAI — how we handle patent search queries, uploaded documents, analysis results, and account data.',
  keywords: [
    'privacy policy', 'CopiioAI privacy', 'patent data protection', 'AI patent privacy',
  ],
  openGraph: {
    title: 'Privacy Policy | CopiioAI',
    description:
      'How CopiioAI handles patent search queries, uploaded documents, analysis results, and account data.',
    url: 'https://copiioai.com/privacy-policy',
    siteName: 'CopiioAI',
    type: 'website',
  },
  alternates: {
    canonical: 'https://copiioai.com/privacy-policy',
  },
}

const COPY = {
  zh: {
    h1: 'CopiioAI 隐私政策',
    intro:
      'CopiioAI 是面向专利检索与分析的 AI 工具。我们尊重您的隐私，并以透明、负责的方式处理数据。本政策说明我们收集哪些数据、如何使用、以及您拥有哪些控制权。',
    sections: [
      {
        h: '1. 我们处理的数据',
        items: [
          '检索提问：您输入的自然语言问题与技术描述。',
          '上传文件：您主动上传的专利说明书、权利要求、审查文件等（PDF / DOCX / XML）。',
          '分析结果：系统为您生成的检索结果、分析与报告。',
          '知识库内容：您在本产品内创建或保存的条目。',
          '账号信息：邮箱地址，以及用于身份验证的令牌。',
          '匿名使用统计：页面访问与功能使用情况、设备与浏览器类型、国家/地区级地理位置。',
        ],
      },
      {
        h: '2. 我们如何使用数据',
        items: [
          '提供并运行 CopiioAI 的核心功能（检索、分析、下载）。',
          '在您的账号下保存对话记录，以便您随时回看。',
          '改进产品可靠性与使用体验。',
        ],
        note: '我们不会将您的数据用于广告、追踪或用户画像。匿名统计仅用于产品改进。',
      },
      {
        h: '3. AI 训练政策',
        p: '您的提问、上传文件与分析结果不会被用于训练任何 AI 模型。模型推理通过企业级 API 通道完成，该通道默认不保留数据用于训练用途。',
      },
      {
        h: '4. 数据安全',
        items: [
          '所有请求经 HTTPS 加密传输。',
          '登录密码以 AES-GCM 加密后传输，服务端不落明文。',
          '对话记录按账号严格隔离，每次读取均按账号过滤。',
          '上传的专利文件在分析处理完成后从服务器删除。',
        ],
        note: '关于每条安全措施的具体说明，请见信息安全页。',
      },
      {
        h: '5. 数据共享',
        p: '我们不会出售或出租您的数据。数据仅在以下情况被处理：为提供核心服务功能而进行的技术处理；您主动选择分享知识库条目时；法律法规要求时。',
      },
      {
        h: '6. 数据留存与您的控制',
        items: [
          '我们仅在提供服务所必需的期间内保留数据。',
          '您可以在产品内删除自己的对话记录与知识库条目。',
          '您可以通过下方邮箱联系我们，请求删除账号数据。',
        ],
      },
      {
        h: '7. 本政策的更新',
        p: '我们可能不定期更新本政策。任何变更都会在本页顶部的更新日期中体现。',
      },
    ],
    contactH: '8. 联系我们',
    contactP: '如对本隐私政策有疑问，请联系：',
    securityLink: '查看信息安全页',
  },
  en: {
    h1: 'Privacy Policy for CopiioAI',
    intro:
      'CopiioAI is an AI-powered patent search and analysis tool. We respect your privacy and handle data transparently and responsibly. This policy explains what we collect, how we use it, and what control you have.',
    sections: [
      {
        h: '1. Data We Process',
        items: [
          'Search queries: the natural-language questions and technical descriptions you enter.',
          'Uploaded files: patent specifications, claims, and prosecution documents you choose to upload (PDF / DOCX / XML).',
          'Analysis results: search results, analyses, and reports generated for you.',
          'Knowledge base entries: items you create or save in the product.',
          'Account information: your email address and an identity token used for authentication.',
          'Anonymous usage statistics: page views and feature usage, device and browser type, country/region-level location.',
        ],
      },
      {
        h: '2. How We Use Data',
        items: [
          'To provide and operate CopiioAI core features (search, analysis, download).',
          'To store conversation records under your account so you can revisit them.',
          'To improve product reliability and user experience.',
        ],
        note: 'We do not use your data for advertising, tracking, or profiling. Anonymous statistics are used solely for product improvement.',
      },
      {
        h: '3. AI Training Policy',
        p: 'Your prompts, uploaded files, and analysis results are never used to train any AI model. Model inference runs through enterprise API channels, which by default do not retain data for training purposes.',
      },
      {
        h: '4. Data Security',
        items: [
          'All requests are transmitted over HTTPS.',
          'Login passwords are encrypted with AES-GCM in transit; no plaintext is stored server-side.',
          'Conversation records are strictly isolated by account, and every read is filtered by account.',
          'Uploaded patent files are deleted from our servers once analysis finishes.',
        ],
        note: 'For a detailed explanation of each measure, see our Information Security page.',
      },
      {
        h: '5. Data Sharing',
        p: 'We do not sell or rent your data. Data is processed only to provide core service functionality, when you choose to share a knowledge base entry, or when required by law.',
      },
      {
        h: '6. Retention and Your Control',
        items: [
          'We retain data only as long as necessary to provide the service.',
          'You may delete your conversation records and knowledge base entries within the product.',
          'You may contact us at the address below to request deletion of your account data.',
        ],
      },
      {
        h: '7. Changes to This Policy',
        p: 'We may update this policy periodically. Any changes will be reflected in the "Last updated" date at the top of this page.',
      },
    ],
    contactH: '8. Contact',
    contactP: 'If you have questions about this Privacy Policy, please contact:',
    securityLink: 'View our Information Security page',
  },
} as const

type CopyLang = keyof typeof COPY

function PolicyCopy({ copy }: { copy: (typeof COPY)[CopyLang] }) {
  return (
    <section className="mb-14">
      <h2 className="text-2xl font-bold text-gray-900 mb-4">{copy.h1}</h2>
      <p className="text-gray-700 mb-8">{copy.intro}</p>

      {copy.sections.map((s) => (
        <div key={s.h}>
          <h3 className="text-xl font-bold text-gray-900 mt-8 mb-3">{s.h}</h3>
          {'p' in s && <p className="text-gray-700 mb-4">{s.p}</p>}
          {'items' in s && (
            <ul className="list-disc pl-6 text-gray-700 mb-4 space-y-1">
              {s.items.map((i) => (
                <li key={i}>{i}</li>
              ))}
            </ul>
          )}
          {'note' in s && <p className="text-gray-600 mb-4">{s.note}</p>}
        </div>
      ))}

      <h3 className="text-xl font-bold text-gray-900 mt-8 mb-3">{copy.contactH}</h3>
      <p className="text-gray-700 mb-2">{copy.contactP}</p>
      <p className="text-gray-700">
        <strong>Email:</strong> copiioai.com@gmail.com
      </p>
    </section>
  )
}

export default function PrivacyPolicy() {
  return (
    <>
      <JsonLd
        id="jsonld-webpage"
        data={{
          '@context': 'https://schema.org',
          '@type': 'WebPage',
          name: 'Privacy Policy',
          description:
            'Privacy Policy for CopiioAI — how we handle patent search queries, uploaded documents, analysis results, and account data.',
          publisher: {
            '@type': 'Organization',
            name: 'CopiioAI',
            url: 'https://copiioai.com',
          },
        }}
      />
      <div className="max-w-3xl mx-auto px-6 py-16">
        <nav className="mb-5 text-sm text-gray-500">
          <a href="/" className="text-teal-600 hover:underline">CopiioAI</a>
          {' / '}
          <span>Privacy Policy</span>
        </nav>

        <h1 className="text-3xl font-bold text-gray-900 mb-2">Privacy Policy</h1>
        <p className="text-gray-500 mb-10">Last updated: {LAST_UPDATED}.</p>

        <PolicyCopy copy={COPY.zh} />
        <div className="border-t border-gray-200 pt-4" />
        <PolicyCopy copy={COPY.en} />

        <div className="mt-12 text-sm">
          <a href="/security" className="text-teal-600 hover:underline">
            {COPY.zh.securityLink} / {COPY.en.securityLink}
          </a>
        </div>
      </div>
    </>
  )
}
```

**注意 TypeScript**：`COPY` 用 `as const`，各 section 的字段（`p` / `items` / `note`）不统一，`'p' in s` 这类收窄在同一定义里可能报错。若 `npm run build` 报类型错误，把 `COPY` 的 `as const` 去掉，并显式声明：

```tsx
interface PolicySection {
  h: string
  p?: string
  items?: readonly string[]
  note?: string
}
interface PolicyCopyShape {
  h1: string
  intro: string
  sections: readonly PolicySection[]
  contactH: string
  contactP: string
  securityLink: string
}
const COPY: Record<'zh' | 'en', PolicyCopyShape> = { /* 同上内容 */ }
```

渲染处相应改为 `{s.p && <p ...>}`、`{s.items && (<ul>...)}`、`{s.note && <p ...>}`。

- [ ] **Step 2: 内容核对（护栏检查）**

逐条比对 `frontend/nextjs/app/(landing)/privacy-policy/page.tsx` 与 §Global Constraints 的护栏清单，确认**六条禁写内容均未出现**。特别检查：

- 没有「自动删除会话内容」类表述（只写「您可以在产品内删除」——这是用户主动操作，属实）
- 没有「不与任何第三方共享」（第 5 节改写为「为提供核心服务功能而进行的技术处理」）
- 没有认证徽章
- 没有厂名

Expected: 六条护栏全部无违反

- [ ] **Step 3: 目视验证**

Run（在 `frontend/nextjs` 下）：`npm run dev`
打开 `http://localhost:3000/privacy-policy`，确认：

1. 页面可达，无「Chrome extension」「turn APIs into AI tools」等旧文案残留
2. 中文段落在上、英文在下
3. 第 3 节写明 AI 训练政策
4. 第 4 节末尾提到信息安全页，且底部链接可达 `/security`
5. 顶部 `Last updated: 2026-09-16.`

Expected: 上述 5 项全部符合

- [ ] **Step 4: 提交**

```bash
git add "frontend/nextjs/app/(landing)/privacy-policy/page.tsx"
git commit -m "docs(privacy): 隐私政策改为专利数据口径并双语化

原文案为旧产品线内容（Chrome 插件采集、开发者工具定位），
与当前 AI 专利检索定位不符，尽调时会引发产品线一致性质疑。"
```

---

### Task 8: 全量验证

**Files:** 无（只验证，不改代码）

**Interfaces:**
- Consumes: Task 1-7 的全部产物
- Produces: 无

- [ ] **Step 1: 单元测试**

Run（在 `frontend/nextjs` 下）：`node --test lib/trustNotice.test.mjs`
Expected: `# pass 4` / `# fail 0`

- [ ] **Step 2: 构建**

Run（在 `frontend/nextjs` 下）：`npm run build`
Expected: 构建成功，无 TypeScript 错误。若报 `app/(landing)/security/page.tsx` 或 `privacy-policy/page.tsx` 的类型错误，按 Task 7 Step 1 的类型声明方案修正后重跑

- [ ] **Step 3: 三处触点的中英文矩阵**

Run（在 `frontend/nextjs` 下）：`npm run dev`
逐格确认，把结果填进下表：

| 位置 | 中文 | 英文 |
|---|---|---|
| 首屏信任卡片（slogan 与输入框之间） | ☐ | ☐ |
| 登录表单底部一行 | ☐ | ☐ |
| 输入框聚焦提示 | ☐ | ☐ |
| 侧边栏「信息安全」入口 | ☐ | ☐ |

Expected: 8 格全部勾选。英文切换通过左下角 / 首屏的语言按钮触发

- [ ] **Step 4: 响应式检查**

在 DevTools 中以 375px 宽刷新：

1. 首屏信任卡片三条承诺竖排、不横向溢出
2. 聚焦输入框 → **不**出现提示（移动端隐藏）
3. 登录表单信任行仍可见

Expected: 上述 3 项全部符合

- [ ] **Step 5: 无动画偏好检查**

DevTools → Rendering → Emulate `prefers-reduced-motion: reduce`，然后聚焦输入框。
Expected: 提示直接出现，无淡入动画

- [ ] **Step 6: 护栏终检**

对以下三个文件全文搜索禁写词：

```bash
cd frontend/nextjs
grep -rniE "自动删除会话|不与任何第三方共享|数据不出境|国内服务器|端到端加密|SOC 2|ISO 27001|等保|佰腾|baiten" \
  lib/trustNotice.js \
  components/app/TrustNotice.tsx \
  "app/(landing)/security/page.tsx" \
  "app/(landing)/privacy-policy/page.tsx"
```

Expected: 无输出

- [ ] **Step 7: 提交（若前述步骤有修正）**

```bash
git add -A frontend/nextjs
git commit -m "fix(security): 全量验证中的修正"
```

若无修正则跳过本步。

---

## 交付后待办（不在本计划范围）

本计划完成后，以下事项仍未做，需另开工作包：

- **W2**：后端日志脱敏与轮转（`api_routes/core.py:728,978,1260`）；`conversations` / `long_tasks` 过期删除；上传目录兜底 GC（覆盖 `celery_worker.py:1349` 之外的崩溃/超时路径）。完成后解锁「会话到期自动删除」，并把 §6 的改写句补进 `/security` 页
- **W3**：审计日志；知识库默认不公开（现 `public=1` 默认他人可读）
- **W4**：等保三级 / ISO 27001 / SOC 2 认证与徽章展示
