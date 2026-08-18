# 聊天页空状态落地布局（Chat Landing）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `/app/chat` 空状态呈现 ChatGPT/DeepSeek 式落地布局——slogan + 居中输入框 + 六大能力（3列×2行）+ 当前可用能力，用户提问后自动切回现有聊天视图。

**Architecture:** 空状态时在 `chat/page.tsx` 中渲染新组件 `ChatLanding`（内含 slogan、复用输入组件 `ChatComposer`、六大能力网格、SceneHint）；有消息时渲染现有消息列表 + 底部输入。`ChatComposer` 从 `page.tsx` 提取，通过 props 接收输入/发送/文件状态（`useChatStream` 的本地 state 不能跨 hook 实例共享，因此必须 props 下发，不能内部调用 hook）。

**Tech Stack:** Next.js 15 App Router（Client Components）、React 18、TypeScript、纯 CSS（popup.css）。无组件测试框架，验证方式为 `npm run build` + 手动检查。

**Spec:** `docs/superpowers/specs/2026-08-18-chat-landing-design.md`（已批准）

## Global Constraints

- 第 ⑥ 张「免费」卡片**不得出现「永久免费」**字样，文案固定为「六大能力当前全部免费开放」/「All six capabilities are currently free」。
- 六大能力卡片**无图标**，纯文字卡片（标题 + 描述）。
- 新文案必须同时加入 `zh.ts` 与 `en.ts`（页面语言切换用现有 `app-i18n`）。
- 仅改动 `/app/chat` 空状态；营销落地页 `/`、结果页 SceneHint、web-app（Vite）一律不动。
- 六大能力网格：桌面 3 列×2 行，平板（<1024px）2 列，手机（<640px）1 列。
- 交互：卡片纯展示不可点；`messages.length === 0` 显示落地布局，第一条消息发送后自动切回聊天视图（`send()` 同步 `setMessages`，无闪烁）。
- 空状态下的文件拖拽/粘贴上传、Enter 发送行为必须保留。

---

### Task 1: i18n 文案（zh + en）

**Files:**
- Modify: `frontend/nextjs/lib/app-i18n/locales/zh.ts`（`chat` 对象内，`welcome` 块之后）
- Modify: `frontend/nextjs/lib/app-i18n/locales/en.ts`（`chat` 对象内，`welcome` 块之后）

**Interfaces:**
- Produces: `t('chat.landing.slogan')`、`t('chat.landing.sectionTitle')`、`t('chat.landing.cap1Title')`…`cap6Title`、`t('chat.landing.cap1Desc')`…`cap6Desc`（`app-i18n` 的 `t()` 支持点路径嵌套查找）

- [ ] **Step 1: 在 zh.ts 的 chat 对象中添加 landing 键**

在 `frontend/nextjs/lib/app-i18n/locales/zh.ts` 中，`welcome: { ... }` 块之后（第 138 行 `prompt: '请输入您的问题，我会尽力帮助您！'` 所在对象的 `},` 后面）插入：

```ts
    landing: {
      slogan: '专利情报，一问即得',
      sectionTitle: '当前可用能力',
      cap1Title: '美国专利检索',
      cap1Desc: '直达 USPTO，按专利号、申请人、技术关键词精准检索美国专利',
      cap2Title: '自然语言检索',
      cap2Desc: '用大白话提问，AI 自动理解技术方案，无需检索式、无需 IPC 分类号',
      cap3Title: '审查历史分析',
      cap3Desc: '自动梳理 OA 答复、修改记录与关键争辩点，生成时间线与风险摘要',
      cap4Title: '跨国同族专利审查历史分析',
      cap4Desc: '透视同族专利全球布局与各国审查历程，多国状态一目了然',
      cap5Title: '全流程文件下载',
      cap5Desc: '专利说明书、权利要求、分析报告，PDF / DOCX 一键下载',
      cap6Title: '免费',
      cap6Desc: '六大能力当前全部免费开放，无需信用卡，无隐藏费用',
    },
```

- [ ] **Step 2: 在 en.ts 的 chat 对象中添加 landing 键**

在 `frontend/nextjs/lib/app-i18n/locales/en.ts` 中同样位置（`welcome` 块之后）插入：

```ts
    landing: {
      slogan: 'Patent intelligence, just ask.',
      sectionTitle: 'Currently available',
      cap1Title: 'US Patent Search',
      cap1Desc: 'Search USPTO directly by patent number, assignee, or technical keywords',
      cap2Title: 'Natural Language Search',
      cap2Desc: 'Ask in plain language — AI understands your intent, no query syntax or IPC classes needed',
      cap3Title: 'Prosecution History Analysis',
      cap3Desc: 'Auto-trace office actions, amendments and key arguments into a timeline with risk summary',
      cap4Title: 'Cross-Border Family Prosecution Analysis',
      cap4Desc: 'See family global layout and per-country prosecution status in one view',
      cap5Title: 'Full-Cycle File Downloads',
      cap5Desc: 'Download patent specs, claims and AI analysis reports as PDF / DOCX',
      cap6Title: 'Free',
      cap6Desc: 'All six capabilities are currently free — no credit card, no hidden fees',
    },
```

- [ ] **Step 3: 校验文案键存在**

Run: `grep -c "landing" frontend/nextjs/lib/app-i18n/locales/zh.ts frontend/nextjs/lib/app-i18n/locales/en.ts`
Expected: 两个文件都输出非零计数（每个文件含 slogan/sectionTitle/6×标题/6×描述共 14 个 landing 相关键）。

- [ ] **Step 4: Commit**

```bash
git add frontend/nextjs/lib/app-i18n/locales/zh.ts frontend/nextjs/lib/app-i18n/locales/en.ts
git commit -m "feat: 聊天落地页中英文文案（slogan/六大能力/小节标题）"
```

---

### Task 2: 提取 ChatComposer 组件（行为保持重构）

**Files:**
- Create: `frontend/nextjs/components/app/ChatComposer.tsx`
- Modify: `frontend/nextjs/app/app/(auth)/chat/page.tsx`

**Interfaces:**
- Consumes: `useChatSession` 的 `input/setInput/streaming`，`useChatStream` 的 `send/abort/selectedFiles/addFiles/removeFile/setIsDragOver`（由 page.tsx 传入 props——`useChatStream` 的 `selectedFiles/isDragOver` 是 hook 本地 state，两个 hook 实例不共享）
- Produces: `export type ChatComposerProps` + `export default function ChatComposer(props: ChatComposerProps)`——渲染 `file-chips-bar`（如有文件）与 `chat-input-wrapper`（附件按钮 + textarea + 发送/停止按钮），不含外层容器

- [ ] **Step 1: 创建 ChatComposer.tsx**

创建 `frontend/nextjs/components/app/ChatComposer.tsx`，内容：

```tsx
'use client'

import { useEffect, useRef } from 'react'
import { useI18n } from '@/lib/app-i18n'

export type ChatComposerProps = {
  input: string
  setInput: (value: string) => void
  streaming: boolean
  send: (files?: File[], presetText?: string) => Promise<void>
  abort: () => void
  selectedFiles: File[]
  addFiles: (files: FileList | File[]) => void
  removeFile: (index: number) => void
  setIsDragOver: (value: boolean) => void
}

export default function ChatComposer({
  input,
  setInput,
  streaming,
  send,
  abort,
  selectedFiles,
  addFiles,
  removeFile,
  setIsDragOver,
}: ChatComposerProps) {
  const { t } = useI18n()
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // Reset the auto-growing textarea height after a send empties the input.
  useEffect(() => {
    if (!input && textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [input])

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
  }

  function handleFilePaste(e: React.ClipboardEvent) {
    const items = e.clipboardData?.files
    if (items && items.length > 0) {
      e.preventDefault()
      addFiles(items)
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files)
    }
  }

  function openFilePicker() {
    fileInputRef.current?.click()
  }

  function getFileTypeBadge(file: File): string {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    if (ext === '.docx') return 'DOCX'
    if (ext === '.xml') return 'XML'
    return 'PDF'
  }

  function formatFileSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  return (
    <>
      {selectedFiles.length > 0 && (
        <div className="file-chips-bar">
          {selectedFiles.map((file, i) => (
            <div key={`${file.name}-${i}`} className="file-chip">
              <span className={`file-chip-badge ${getFileTypeBadge(file).toLowerCase()}`}>
                {getFileTypeBadge(file)}
              </span>
              <span className="file-chip-name">{file.name}</span>
              <span className="file-chip-size">{formatFileSize(file.size)}</span>
              <button
                className="file-chip-remove"
                onClick={() => removeFile(i)}
                aria-label="Remove file"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}
      <div
        className="chat-input-wrapper"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="file-input-hidden"
          accept=".pdf,.docx,.xml,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/xml,text/xml"
          multiple
          onChange={e => { if (e.target.files) addFiles(e.target.files); e.target.value = '' }}
        />
        <button
          className="file-upload-btn"
          onClick={openFilePicker}
          aria-label="Attach patent files"
          title={t('chat.attachFiles') || 'Attach patent specification files (PDF, DOCX, XML)'}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
          </svg>
        </button>
        <textarea
          ref={textareaRef}
          className="chat-input"
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          onPaste={handleFilePaste}
          placeholder={t('chat.placeholder')}
          rows={1}
        />
        {streaming ? (
          <button
            className="send-btn"
            onClick={abort}
            style={{ background: 'var(--color-text-secondary)' }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="6" width="12" height="12" />
            </svg>
          </button>
        ) : (
          <button
            className="send-btn"
            onClick={() => send()}
            disabled={!input.trim() && selectedFiles.length === 0}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        )}
      </div>
    </>
  )
}
```

- [ ] **Step 2: 在 page.tsx 中删除已迁移的 refs、handlers 与 effect**

在 `frontend/nextjs/app/app/(auth)/chat/page.tsx` 中删除：
- 第 48-49 行：`textareaRef`、`fileInputRef` 两个 ref 声明
- 第 53-60 行：重置 textarea 高度的 `useEffect`（含注释块）
- 第 295-352 行：`handleKeyDown`、`handleInput`、`handleFilePaste`、`handleDragOver`、`handleDragLeave`、`handleDrop`、`openFilePicker`、`getFileTypeBadge`、`formatFileSize` 共 9 个函数

保留 `handleDragOver/handleDragLeave/handleDrop` 在 page.tsx 中的副本（第 400-403 行的 `file-drop-overlay` 仍在使用它们——覆盖层与输入框的拖拽处理各自独立）。

- [ ] **Step 3: 用 ChatComposer 替换 page.tsx 中 composer JSX**

在 `frontend/nextjs/app/app/(auth)/chat/page.tsx`：
1. 导入：`import ChatComposer from '@/components/app/ChatComposer'`
2. 将第 417-495 行的整个 composer JSX 块（从 `{selectedFiles.length > 0 && (` 到 `</div>` 结束的 `chat-input-wrapper` 闭合）替换为：

```tsx
          <ChatComposer
            input={input}
            setInput={setInput}
            streaming={streaming}
            send={send}
            abort={abort}
            selectedFiles={selectedFiles}
            addFiles={addFiles}
            removeFile={removeFile}
            setIsDragOver={setIsDragOver}
          />
```

（外层 `.chat-input-container` 容器保留不动，ChatComposer 只渲染内部内容。）

- [ ] **Step 4: 构建验证**

Run（在 `frontend/nextjs` 目录）: `npm run build`
Expected: 构建成功，无 TypeScript 错误。

- [ ] **Step 5: 手动验证（行为保持）**

`npm run dev` 后登录进入 `/app/chat`：
- 输入框在底部，可正常输入/Enter 发送
- 附件按钮、拖拽文件显示 file-drop-overlay、粘贴文件生成 file-chip、删除 chip 均正常
- streaming 时发送按钮变为停止按钮

- [ ] **Step 6: Commit**

```bash
git add frontend/nextjs/components/app/ChatComposer.tsx "frontend/nextjs/app/app/(auth)/chat/page.tsx"
git commit -m "refactor: 提取 ChatComposer 组件（输入区行为保持不变）"
```

---

### Task 3: ChatLanding 组件 + CSS + 空状态集成

**Files:**
- Create: `frontend/nextjs/components/app/ChatLanding.tsx`
- Modify: `frontend/nextjs/app/app/(auth)/chat/page.tsx`
- Modify: `frontend/nextjs/styles/popup.css`（文件末尾追加）

**Interfaces:**
- Consumes: `ChatComposerProps`（Task 2 定义）、`t('chat.landing.*')`（Task 1 定义）、现有 `SceneHint` 组件
- Produces: `export default function ChatLanding(composerProps: ChatComposerProps)`；CSS 类 `.chat-landing` / `.chat-landing-slogan` / `.chat-landing-composer` / `.chat-landing-grid` / `.chat-landing-card`（`.free` 修饰）/ `.chat-landing-section` / `.chat-landing-section-title`

- [ ] **Step 1: 创建 ChatLanding.tsx**

创建 `frontend/nextjs/components/app/ChatLanding.tsx`：

```tsx
'use client'

import { useI18n } from '@/lib/app-i18n'
import ChatComposer, { type ChatComposerProps } from './ChatComposer'
import SceneHint from './SceneHint'

const CAPABILITIES = [
  { titleKey: 'chat.landing.cap1Title', descKey: 'chat.landing.cap1Desc', free: false },
  { titleKey: 'chat.landing.cap2Title', descKey: 'chat.landing.cap2Desc', free: false },
  { titleKey: 'chat.landing.cap3Title', descKey: 'chat.landing.cap3Desc', free: false },
  { titleKey: 'chat.landing.cap4Title', descKey: 'chat.landing.cap4Desc', free: false },
  { titleKey: 'chat.landing.cap5Title', descKey: 'chat.landing.cap5Desc', free: false },
  { titleKey: 'chat.landing.cap6Title', descKey: 'chat.landing.cap6Desc', free: true },
] as const

export default function ChatLanding(composerProps: ChatComposerProps) {
  const { t } = useI18n()

  return (
    <div className="chat-landing">
      <h2 className="chat-landing-slogan">{t('chat.landing.slogan')}</h2>
      <div className="chat-landing-composer">
        <ChatComposer {...composerProps} />
      </div>
      <div className="chat-landing-grid">
        {CAPABILITIES.map((cap) => (
          <div key={cap.titleKey} className={`chat-landing-card${cap.free ? ' free' : ''}`}>
            <h3 className="chat-landing-card-title">{t(cap.titleKey)}</h3>
            <p className="chat-landing-card-desc">{t(cap.descKey)}</p>
          </div>
        ))}
      </div>
      <div className="chat-landing-section">
        <h3 className="chat-landing-section-title">{t('chat.landing.sectionTitle')}</h3>
        <SceneHint />
      </div>
    </div>
  )
}
```

- [ ] **Step 2: 追加 CSS 到 popup.css**

在 `frontend/nextjs/styles/popup.css` 文件末尾追加：

```css
/* =============================================================================
   Chat Landing (empty state) — slogan + centered composer + six capabilities
   ============================================================================= */

.chat-landing {
  width: 100%;
  max-width: var(--chat-max-width);
  margin: 0 auto;
  min-height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: var(--spacing-6);
  padding: var(--spacing-8) var(--spacing-4);
  animation: fadeIn 0.3s ease-out;
}

.chat-landing-slogan {
  font-size: 32px;
  font-weight: 700;
  color: var(--color-text-primary);
  text-align: center;
  line-height: 1.3;
}

.chat-landing-composer {
  width: 100%;
  max-width: 720px;
  padding: 0 var(--spacing-4);
}

.chat-landing-grid {
  width: 100%;
  max-width: 900px;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--spacing-3);
  padding: 0 var(--spacing-4);
}

.chat-landing-card {
  background: var(--color-bg-white);
  border: 1px solid var(--color-border);
  border-radius: 16px;
  padding: var(--spacing-4);
  transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
}

.chat-landing-card:hover {
  transform: translateY(-2px);
  border-color: var(--color-primary);
  box-shadow: var(--shadow-sm);
}

.chat-landing-card.free {
  border-color: #f59e0b;
}

.chat-landing-card.free .chat-landing-card-title {
  color: #b45309;
}

.chat-landing-card-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: var(--spacing-1);
}

.chat-landing-card-desc {
  font-size: 13px;
  color: var(--color-text-secondary);
  line-height: 1.6;
}

.chat-landing-section {
  width: 100%;
  max-width: 900px;
  padding: 0 var(--spacing-4);
}

.chat-landing-section-title {
  font-size: 14px;
  font-weight: 500;
  color: var(--color-text-secondary);
  text-align: center;
  margin-bottom: var(--spacing-3);
}

@media (max-width: 1023px) {
  .chat-landing-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 639px) {
  .chat-landing-grid {
    grid-template-columns: 1fr;
  }
  .chat-landing-slogan {
    font-size: 24px;
  }
}
```

- [ ] **Step 3: page.tsx 集成空状态切换**

在 `frontend/nextjs/app/app/(auth)/chat/page.tsx`：
1. 导入：`import ChatLanding from '@/components/app/ChatLanding'`
2. 在组件顶部（`const bottomRef` 声明附近）加：`const showLanding = messages.length === 0`
3. 将 `.chat-messages` 内的空状态 JSX（第 358-365 行的 `{messages.length === 0 && ( ... welcome empty-state ... )}`）和其后的 `<SceneHint />`（第 366 行）整体替换为：

```tsx
          {showLanding ? (
            <ChatLanding
              input={input}
              setInput={setInput}
              streaming={streaming}
              send={send}
              abort={abort}
              selectedFiles={selectedFiles}
              addFiles={addFiles}
              removeFile={removeFile}
              setIsDragOver={setIsDragOver}
            />
          ) : (
            <>
              {messages.map((msg) => (
                /* 现有消息渲染块原样保留 */
              ))}
              <div ref={bottomRef} />
            </>
          )}
```

4. 将第 416 行的 `.chat-input-container` 块改为仅在非落地模式渲染，并同样替换为 ChatComposer：

```tsx
        {!showLanding && (
          <div className="chat-input-container">
            <ChatComposer
              input={input}
              setInput={setInput}
              streaming={streaming}
              send={send}
              abort={abort}
              selectedFiles={selectedFiles}
              addFiles={addFiles}
              removeFile={removeFile}
              setIsDragOver={setIsDragOver}
            />
          </div>
        )}
```

5. 删除 `import SceneHint from '@/components/app/SceneHint'`（已移入 ChatLanding 内部，page.tsx 不再直接使用）

- [ ] **Step 4: 构建验证**

Run（在 `frontend/nextjs` 目录）: `npm run build`
Expected: 构建成功，无 TypeScript 错误。

- [ ] **Step 5: 手动验证**

`npm run dev` 后登录进入 `/app/chat`（空会话）：
- 屏幕中央显示 slogan「专利情报，一问即得」+ 居中输入框 + 六大能力（桌面 3 列×2 行，无图标）+「当前可用能力」小节标题 + 场景能力列表
- 语言切换到 EN：slogan 变为 "Patent intelligence, just ask."，六大能力与小节标题同步切换
- 免费卡片标题为琥珀色；文案含「当前全部免费开放」，不含「永久免费」
- 在居中输入框输入并发送 → 立即切回消息列表 + 底部输入框布局
- 空状态下拖拽文件 → 显示 file-drop-overlay；粘贴文件 → 生成 file-chip；Enter 发送正常
- 缩窄窗口至 <1024px：卡片变 2 列；<640px：变 1 列，slogan 缩小
- 结果页 SceneHint 与营销落地页 `/` 无任何变化

- [ ] **Step 6: Commit**

```bash
git add frontend/nextjs/components/app/ChatLanding.tsx "frontend/nextjs/app/app/(auth)/chat/page.tsx" frontend/nextjs/styles/popup.css
git commit -m "feat: 聊天页空状态落地布局（slogan + 居中输入框 + 六大能力 + 可用能力）"
```

---

## Self-Review

- **Spec 覆盖：** 布局结构（Task 3）✓；六大能力 3×2 + 响应式（Task 3 CSS）✓；中英文文案含免费措辞约束（Task 1）✓；slogan 已定文案（Task 1）✓；ChatComposer 提取（Task 2）✓；SceneHint 移到六大能力下方（Task 3 Step 3 第 3 点 + ChatLanding）✓；无图标（Task 3 组件与 CSS 均无 icon）✓；结果页/营销页不动（Task 3 不触碰相关文件）✓。
- **占位符扫描：** 无 TBD/TODO；Task 3 Step 3 第 3 点的消息渲染块是"原样保留"引用——指向当前 page.tsx 中存在的 `messages.map` 块（第 367-394 行），实施者在其当前文件中直接可见，非占位符。
- **类型一致性：** `ChatComposerProps` 在 Task 2 定义并导出，Task 2 Step 3 与 Task 3 Step 3 传入的 props 名称完全一致（input/setInput/streaming/send/abort/selectedFiles/addFiles/removeFile/setIsDragOver）；`t()` 键名与 Task 1 定义的键一一对应。
