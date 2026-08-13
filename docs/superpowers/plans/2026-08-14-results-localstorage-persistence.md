# 检索结果浏览器本地持久化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 检索结果集写入浏览器 localStorage（裁剪版、上限 100 集），结果页/聊天页刷新后从 localStorage 恢复，存储不可用时静默降级回内存模式。

**Architecture:** 新增纯函数模块 `lib/resultsStore.js`（storage 注入、node 可测）；`useChatStream` 在 json artifact 解码成功后立即持久化；结果页三层解析（内存精确 → localStorage 合成消息 → 内存最新兜底）；聊天页/结果页水合后按 queryText 匹配恢复结果卡片。SSR 安全：渲染路径 store 用两阶段 state 读取，effect 路径现读现用。

**Tech Stack:** Next.js App Router（静态导出）+ 纯函数 node 测试（`node --test`）。

**Spec:** `docs/superpowers/specs/2026-08-14-results-localstorage-persistence-design.md`

## Global Constraints

- 分支：`feature/search-results-split-view`（当前 HEAD `fa0154e`）
- 前端测试命令（cwd `frontend/nextjs`）：`node --test lib/*.test.mjs`；`npx tsc --noEmit`；`npm run build`（三选全跑）
- 每任务末尾独立 commit（conventional commits，无 Co-Authored-By 尾注）
- 不可变更新（返回新对象，不修改入参）；无 console.log；遵循周边代码风格
- localStorage 访问必须客户端侧 + try/catch 静默降级；渲染路径禁止直接读 storage（水合安全）
- 存储键名固定 `copiioai_results`；上限 100 集、index 上限 200 条
- 后端零改动

---

### Task 1: `lib/resultsStore.js` 核心模块

**Files:**
- Create: `frontend/nextjs/lib/resultsStore.js`
- Create: `frontend/nextjs/lib/resultsStore.test.mjs`

**Interfaces:**
- Consumes: `pruneResultsForPersistence`（`lib/results.js`，已存在，签名 `(results, {maxRows, abstractLimit}) => prunedResults`）
- Produces（Task 2/3/4 依赖）：
  - `loadResultsStore(storage?)` → `{ sets: { [setId]: { source, columns, rows } }, index: [{ setId, sessionId, queryText, savedAt }] }`（storage 缺失/异常 → 空 store）
  - `persistResultsSet(store, results, meta)` → 新 store（不可变）；results 经 prune 裁剪后写入 `sets[results.setId]`；index 头部追加 meta；sets 超 100 丢最旧（对象插入序）
  - `saveResultsStore(storage, store)` → 落盘；失败时从最旧 set 起逐个丢弃重试；全空仍失败 → 静默放弃
  - `persistResultsSetToStorage(storage, results, meta)` → `saveResultsStore(storage, persistResultsSet(loadResultsStore(storage), results, meta))` 一步便捷函数
  - `restoreResultsInMessages(messages, store)` → 新 messages（不可变）：index 从新到旧扫描，`queryText` 命中 role==='user' 的消息 → 紧随其后的消息若无 results 且 `store.sets[setId]` 存在 → 挂 `results: { setId, ...set }`
  - `buildStoredMessage(setId, store)` → `{ id: 'stored-' + setId, role: 'assistant', content: '', results: { setId, ...set } }` | `null`

- [ ] **Step 1: 写失败测试**

创建 `frontend/nextjs/lib/resultsStore.test.mjs`：

```js
import test from 'node:test'
import assert from 'node:assert/strict'

import {
  loadResultsStore,
  persistResultsSet,
  saveResultsStore,
  persistResultsSetToStorage,
  restoreResultsInMessages,
  buildStoredMessage,
} from './resultsStore.js'

const RESULTS = (setId, rows = 10) => ({
  setId,
  source: 'uspto',
  columns: [
    { key: 'title', label: '标题', role: 'title' },
    { key: 'abstractText', label: '摘要', role: 'abstract' },
    { key: 'eventDataBag', label: '事件', role: 'text' },
  ],
  rows: Array.from({ length: rows }, (_, i) => ({
    title: `专利 ${i}`,
    abstractText: '字'.repeat(600),
    eventDataBag: 'x'.repeat(1000),
  })),
})

function fakeStorage(initial = {}) {
  const map = new Map(Object.entries(initial))
  return {
    getItem: (key) => (map.has(key) ? map.get(key) : null),
    setItem: (key, value) => { map.set(key, value) },
    _map: map,
  }
}

test('persistResultsSet prunes rows and drops text-role columns', () => {
  const store = persistResultsSet(loadResultsStore(null), RESULTS('s1', 60), {
    setId: 's1', sessionId: null, queryText: '找专利', savedAt: 1,
  })
  assert.equal(store.sets.s1.rows.length, 50) // prune 上限 50 行
  assert.ok(store.sets.s1.rows[0].abstractText.length <= 500)
  assert.equal('eventDataBag' in store.sets.s1.rows[0], false) // text 列被丢
  assert.equal('eventDataBag' in store.sets.s1.columns, false)
})

test('persistResultsSet prepends index and caps sets at 100 dropping oldest', () => {
  let store = loadResultsStore(null)
  for (let i = 0; i < 103; i += 1) {
    store = persistResultsSet(store, RESULTS(`s${i}`), {
      setId: `s${i}`, sessionId: null, queryText: `q${i}`, savedAt: i,
    })
  }
  assert.equal(Object.keys(store.sets).length, 100)
  assert.equal('s0' in store.sets, false) // 最旧的 3 个被丢
  assert.equal('s3' in store.sets, true)
  assert.equal(store.index[0].setId, 's102') // 最新在前
  assert.equal(store.index[0].queryText, 'q102')
})

test('saveResultsStore drops oldest sets when quota is exceeded', () => {
  const storage = fakeStorage()
  let store = persistResultsSet(loadResultsStore(null), RESULTS('a'), {
    setId: 'a', sessionId: null, queryText: 'qa', savedAt: 1,
  })
  store = persistResultsSet(store, RESULTS('b'), {
    setId: 'b', sessionId: null, queryText: 'qb', savedAt: 2,
  })
  let attempts = 0
  storage.setItem = (key, value) => {
    attempts += 1
    if (attempts < 2) throw new Error('QuotaExceededError')
    storage._map.set(key, value)
  }
  saveResultsStore(storage, store) // 第 1 次失败丢 a，第 2 次只剩 {b} 成功
  const saved = JSON.parse(storage._map.get('copiioai_results'))
  assert.equal('a' in saved.sets, false)
  assert.equal('b' in saved.sets, true)
})

test('saveResultsStore gives up silently when storage is unusable', () => {
  const storage = { getItem: () => null, setItem: () => { throw new Error('SecurityError') } }
  const store = persistResultsSet(loadResultsStore(null), RESULTS('a'), {
    setId: 'a', sessionId: null, queryText: 'qa', savedAt: 1,
  })
  saveResultsStore(storage, store) // 不抛异常
})

test('persistResultsSetToStorage loads, persists, and saves', () => {
  const storage = fakeStorage()
  persistResultsSetToStorage(storage, RESULTS('s1'), {
    setId: 's1', sessionId: 'sid-1', queryText: '找专利', savedAt: 10,
  })
  const saved = JSON.parse(storage._map.get('copiioai_results'))
  assert.ok(saved.sets.s1)
  assert.equal(saved.index[0].queryText, '找专利')
})

test('restoreResultsInMessages attaches results after matching user message', () => {
  const store = persistResultsSet(loadResultsStore(null), RESULTS('s1'), {
    setId: 's1', sessionId: null, queryText: '找专利', savedAt: 1,
  })
  const messages = [
    { id: 'u1', role: 'user', content: '找专利' },
    { id: 'a1', role: 'assistant', content: '找到了 10 条' },
  ]
  const restored = restoreResultsInMessages(messages, store)
  assert.equal(restored[0].results, undefined)
  assert.equal(restored[1].results.setId, 's1')
  assert.equal(restored[1].results.rows.length, 10)
})

test('restoreResultsInMessages skips non-matching and already-restored messages', () => {
  const store = persistResultsSet(loadResultsStore(null), RESULTS('s1'), {
    setId: 's1', sessionId: null, queryText: '找专利', savedAt: 1,
  })
  const messages = [
    { id: 'u1', role: 'user', content: '别的提问' },
    { id: 'a1', role: 'assistant', content: 'x' },
    { id: 'u2', role: 'user', content: '找专利' },
    { id: 'a2', role: 'assistant', content: 'y', results: { setId: 'existing' } },
  ]
  const restored = restoreResultsInMessages(messages, store)
  assert.equal(restored[1].results, undefined)
  assert.equal(restored[3].results.setId, 'existing') // 已有 results 不动
})

test('restoreResultsInMessages uses most recent index entry for duplicate queryText', () => {
  let store = persistResultsSet(loadResultsStore(null), RESULTS('old'), {
    setId: 'old', sessionId: null, queryText: '找专利', savedAt: 1,
  })
  store = persistResultsSet(store, RESULTS('new'), {
    setId: 'new', sessionId: null, queryText: '找专利', savedAt: 2,
  })
  const messages = [
    { id: 'u1', role: 'user', content: '找专利' },
    { id: 'a1', role: 'assistant', content: 'x' },
  ]
  const restored = restoreResultsInMessages(messages, store)
  assert.equal(restored[1].results.setId, 'new') // 最新 index 条目获胜
})

test('buildStoredMessage builds synthetic assistant message or null', () => {
  const store = persistResultsSet(loadResultsStore(null), RESULTS('s1'), {
    setId: 's1', sessionId: null, queryText: 'q', savedAt: 1,
  })
  assert.equal(buildStoredMessage('s1', store).id, 'stored-s1')
  assert.equal(buildStoredMessage('s1', store).results.setId, 's1')
  assert.equal(buildStoredMessage('missing', store), null)
  assert.equal(buildStoredMessage('s1', null), null)
})

test('loadResultsStore tolerates missing storage and corrupted JSON', () => {
  assert.deepEqual(loadResultsStore(null), { sets: {}, index: [] })
  const bad = fakeStorage({ copiioai_results: '{not json' })
  assert.deepEqual(loadResultsStore(bad), { sets: {}, index: [] })
  const wrongShape = fakeStorage({ copiioai_results: '{"sets": "nope"}' })
  assert.deepEqual(loadResultsStore(wrongShape), { sets: {}, index: [] })
})
```

- [ ] **Step 2: 运行确认失败**

Run: `node --test lib/resultsStore.test.mjs`（cwd `frontend/nextjs`）
Expected: FAIL —— `Cannot find module './resultsStore.js'`

- [ ] **Step 3: 实现**

创建 `frontend/nextjs/lib/resultsStore.js`：

```js
/**
 * Browser-local persistence for patent search result sets.
 * Result sets are pruned copies of the decoded format=json artifact,
 * kept in localStorage under one key.  All storage access is guarded —
 * unavailable storage (private mode, disabled site data) degrades to
 * the in-memory behaviour silently.
 */

import { pruneResultsForPersistence } from './results.js'

export const RESULTS_STORE_KEY = 'copiioai_results'
export const MAX_RESULT_SETS = 100
export const MAX_INDEX_ENTRIES = 200

export function emptyResultsStore() {
  return { sets: {}, index: [] }
}

export function loadResultsStore(storage) {
  if (!storage) return emptyResultsStore()
  try {
    const parsed = JSON.parse(storage.getItem(RESULTS_STORE_KEY))
    if (
      !parsed
      || typeof parsed !== 'object'
      || typeof parsed.sets !== 'object'
      || parsed.sets === null
      || !Array.isArray(parsed.index)
    ) {
      return emptyResultsStore()
    }
    return parsed
  } catch {
    return emptyResultsStore()
  }
}

function dropOldestSet(store) {
  const keys = Object.keys(store.sets)
  if (keys.length === 0) return store
  const { [keys[0]]: _dropped, ...rest } = store.sets
  return { ...store, sets: rest }
}

export function persistResultsSet(store, results, meta) {
  const pruned = pruneResultsForPersistence(results)
  let sets = { ...(store?.sets || {}), [results.setId]: pruned }
  const keys = Object.keys(sets)
  while (keys.length > MAX_RESULT_SETS) {
    delete sets[keys[0]]
    keys.shift()
  }
  const index = [
    {
      setId: results.setId,
      sessionId: meta.sessionId ?? null,
      queryText: meta.queryText ?? '',
      savedAt: meta.savedAt ?? 0,
    },
    ...(store?.index || []),
  ].slice(0, MAX_INDEX_ENTRIES)
  return { sets, index }
}

export function saveResultsStore(storage, store) {
  if (!storage) return
  let current = store
  while (true) {
    try {
      storage.setItem(RESULTS_STORE_KEY, JSON.stringify(current))
      return
    } catch {
      const next = dropOldestSet(current)
      if (Object.keys(next.sets).length === Object.keys(current.sets).length) {
        return // 没有可丢的了（或错误不是配额导致）——静默放弃
      }
      current = next
    }
  }
}

export function persistResultsSetToStorage(storage, results, meta) {
  saveResultsStore(storage, persistResultsSet(loadResultsStore(storage), results, meta))
}

export function restoreResultsInMessages(messages, store) {
  if (!Array.isArray(messages) || !store || !Array.isArray(store.index)) return messages
  const sets = store.sets || {}
  return messages.map((msg, index) => {
    if (msg.role !== 'assistant' || msg.results || index === 0) return msg
    const previous = messages[index - 1]
    if (!previous || previous.role !== 'user') return msg
    for (const entry of store.index) {
      if (entry.queryText && entry.queryText === previous.content && sets[entry.setId]) {
        return { ...msg, results: { setId: entry.setId, ...sets[entry.setId] } }
      }
    }
    return msg
  })
}

export function buildStoredMessage(setId, store) {
  if (!store || !setId) return null
  const set = store.sets && store.sets[setId]
  if (!set) return null
  return {
    id: `stored-${setId}`,
    role: 'assistant',
    content: '',
    results: { setId, ...set },
  }
}
```

- [ ] **Step 4: 运行确认通过**

Run: `node --test lib/resultsStore.test.mjs`
Expected: 11/11 PASS

- [ ] **Step 5: Commit**

```bash
git add lib/resultsStore.js lib/resultsStore.test.mjs
git commit -m "feat: browser-local results persistence store module"
```

---

### Task 2: `resolveActiveResultsMessage` 增加 store 层

**Files:**
- Modify: `frontend/nextjs/lib/results.js`（`resolveActiveResultsMessage`）
- Modify: `frontend/nextjs/lib/results.test.mjs`

**Interfaces:**
- Consumes: `buildStoredMessage(setId, store)`（Task 1）
- Produces: `resolveActiveResultsMessage(messages, urlSetId, store?)` —— store 为可选第三参（既有调用不破坏）。查找顺序：内存精确匹配 → store 按 setId 合成消息 → 内存最新兜底。

- [ ] **Step 1: 写失败测试**

在 `frontend/nextjs/lib/results.test.mjs` 末尾追加：

```js
test('resolveActiveResultsMessage falls back to stored set before newest fallback', () => {
  const messages = [RESULT_MESSAGE('set-a')]
  const store = { sets: { 'stored-x': { source: 'uspto', columns: [], rows: [] } }, index: [] }
  const resolved = resolveActiveResultsMessage(messages, 'stored-x', store)
  assert.equal(resolved.id, 'stored-stored-x')
  assert.equal(resolved.results.setId, 'stored-x')
})

test('resolveActiveResultsMessage prefers exact memory match over store', () => {
  const messages = [RESULT_MESSAGE('set-a')]
  const store = { sets: { 'set-a': { source: 'uspto', columns: [], rows: [] } }, index: [] }
  assert.equal(resolveActiveResultsMessage(messages, 'set-a', store).id, 'm-set-a')
})

test('resolveActiveResultsMessage keeps newest fallback when store misses', () => {
  const messages = [RESULT_MESSAGE('set-a')]
  const store = { sets: {}, index: [] }
  assert.equal(resolveActiveResultsMessage(messages, 'missing', store).id, 'm-set-a')
})
```

- [ ] **Step 2: 运行确认失败**

Run: `node --test lib/results.test.mjs`
Expected: 前两个新测试 FAIL（store 参数未生效/不存在）

- [ ] **Step 3: 实现**

修改 `frontend/nextjs/lib/results.js` 中 `resolveActiveResultsMessage`：

```js
export function resolveActiveResultsMessage(messages, urlSetId, store) {
  const list = Array.isArray(messages) ? messages : []
  let newest = null
  for (const message of list) {
    if (!message || !message.results) continue
    if (urlSetId && message.results.setId === urlSetId) return message
    newest = message
  }
  if (urlSetId && store) {
    const storedMessage = buildStoredMessage(urlSetId, store)
    if (storedMessage) return storedMessage
  }
  return newest || undefined
}
```

文件顶部新增 import：

```js
import { buildStoredMessage } from './resultsStore.js'
```

（`results.js` 与 `resultsStore.js` 互相依赖：resultsStore imports pruneResultsForPersistence from results.js，results.js imports buildStoredMessage from resultsStore.js —— ESM 循环引用在纯函数场景下安全：两个模块顶层不执行对方函数，仅绑定引用。如 node --test 报错，改为在 resolveActiveResultsMessage 内动态 `import('./resultsStore.js')` 不可行（同步函数），替代方案是把 buildStoredMessage 逻辑内联或用 `store.sets` 直接构造。优先尝试顶层 import。）

- [ ] **Step 4: 运行确认通过**

Run: `node --test lib/results.test.mjs lib/resultsStore.test.mjs`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add lib/results.js lib/results.test.mjs
git commit -m "feat: resolve results from localStorage store in results page"
```

---

### Task 3: `useChatStream` 流结束后持久化

**Files:**
- Modify: `frontend/nextjs/lib/useChatStream.ts`（json artifact 解码处 + 持久化调用）

**Interfaces:**
- Consumes: `persistResultsSetToStorage`、`decodeArtifactChunksToResults`、`pruneResultsForPersistence`（不直接用——persist 内部处理）
- Produces: 无新导出；持久化副作用在 `send()` 的 artifact_end 分支内完成

- [ ] **Step 1: 修改解码处保留完整结果**

`frontend/nextjs/lib/useChatStream.ts` 的 SSE 循环中，`if (event.type === 'artifact_end')` 分支内，把当前代码：

```ts
            if (event.type === 'artifact_end') {
              const endArtifactId = String(event.artifact_id ?? event.artifactId ?? '')
              if (pendingJsonId !== null && endArtifactId === pendingJsonId) {
                decodedSetId = decodeArtifactChunksToResults(
                  pendingJsonChunks, pendingJsonId,
                )?.setId ?? null
                pendingJsonId = null
              }
```

改为：

```ts
            if (event.type === 'artifact_end') {
              const endArtifactId = String(event.artifact_id ?? event.artifactId ?? '')
              if (pendingJsonId !== null && endArtifactId === pendingJsonId) {
                const decodedResults = decodeArtifactChunksToResults(
                  pendingJsonChunks, pendingJsonId,
                )
                decodedSetId = decodedResults?.setId ?? null
                pendingJsonId = null
                // Persist a pruned copy to browser localStorage so the
                // results survive refresh / tab reopen.  Unavailable
                // storage degrades silently (no-op).
                if (decodedResults) {
                  persistResultsSetToStorage(
                    window.localStorage,
                    decodedResults,
                    { sessionId, queryText: text, savedAt: Date.now() },
                  )
                }
              }
```

（`text` 与 `sessionId` 是 `send()` 闭包内已有变量；`decodedSetId` 声明处（`let decodedSetId: string | null = null`）不变。）

文件顶部新增 import：

```ts
import { persistResultsSetToStorage } from '@/lib/resultsStore'
```

- [ ] **Step 2: 验证编译与测试**

Run（cwd `frontend/nextjs`）：
1. `npx tsc --noEmit` — Expected: 无输出、退出码 0
2. `node --test lib/*.test.mjs` — Expected: 全部 PASS（既有测试不回归）

- [ ] **Step 3: Commit**

```bash
git add lib/useChatStream.ts
git commit -m "feat: persist pruned results to localStorage after streaming"
```

---

### Task 4: 结果页 + 聊天页恢复接线

**Files:**
- Modify: `frontend/nextjs/app/app/(auth)/results/page.tsx`
- Modify: `frontend/nextjs/app/app/(auth)/chat/page.tsx`

**Interfaces:**
- Consumes: `loadResultsStore`、`restoreResultsInMessages`（Task 1）、`resolveActiveResultsMessage(messages, setId, store)`（Task 2）
- Produces: 无新导出

- [ ] **Step 1: 结果页三层解析（两阶段 store 读取）**

`frontend/nextjs/app/app/(auth)/results/page.tsx`：

1. import 行（`@/lib/results` 处）：

```ts
import { buildRowModel, resolveActiveResultsMessage } from '@/lib/results'
import { loadResultsStore, restoreResultsInMessages } from '@/lib/resultsStore'
```

2. 组件内 `activeMessage` useMemo 之前，加 store 状态：

```ts
  // Two-phase store read: render-path resolution must not touch
  // localStorage during SSR/hydration (static export).
  const [resultsStore, setResultsStore] = useState<ReturnType<typeof loadResultsStore> | null>(null)

  useEffect(() => {
    setResultsStore(loadResultsStore(window.localStorage))
  }, [])
```

3. `activeMessage` useMemo 改为传入 store：

```ts
  const activeMessage: ChatMessage | undefined = useMemo(() => {
    return resolveActiveResultsMessage(messages, setId, resultsStore) || undefined
  }, [messages, setId, resultsStore])
```

- [ ] **Step 2: 结果页水合恢复**

结果页现有 hydrate effect 中 `setMessages(data.messages ...)` 映射处（`loaded` 数组构建完成后）改为应用恢复：

```ts
        const loaded = data.messages
          .filter((m: any) => m.role && m.content)
          .map(/* …既有映射不动… */)
        setMessages(
          restoreResultsInMessages(loaded, loadResultsStore(window.localStorage)),
        )
```

（原代码是 `setMessages(data.messages.filter(...).map(...))` —— 把 `.filter().map()` 的结果先存 `loaded` 变量再包一层 `restoreResultsInMessages`，其余逻辑一行不动。）

- [ ] **Step 3: 聊天页水合恢复**

`frontend/nextjs/app/app/(auth)/chat/page.tsx` 的会话加载 effect（getSession 成功分支）中，`setMessages(loaded)` 处同样包一层。具体位置：effect 内 `data.messages` 过滤映射后调用 setMessages 的那一行，改为：

```ts
          const loaded = data.messages
            .filter(/* 既有条件不动 */)
            .map(/* 既有映射不动 */)
          setMessages(restoreResultsInMessages(loaded, loadResultsStore(window.localStorage)))
```

并在该文件顶部 import：

```ts
import { loadResultsStore, restoreResultsInMessages } from '@/lib/resultsStore'
```

**注意**：先 Read 该文件 effect 区域确认确切的 setMessages 调用形态，按实际代码适配（不改过滤/映射逻辑本身）。

- [ ] **Step 4: 验证**

Run（cwd `frontend/nextjs`）：
1. `node --test lib/*.test.mjs` — Expected: 全部 PASS
2. `npx tsc --noEmit` — Expected: 无输出、退出码 0
3. `npm run build` — Expected: 构建成功

- [ ] **Step 5: Commit**

```bash
git add "app/app/(auth)/results/page.tsx" "app/app/(auth)/chat/page.tsx"
git commit -m "feat: restore results from localStorage on session hydration"
```

---

## 手动验证清单（测试环境，仅前端部署）

```bash
# 服务器
git pull
cd frontend/nextjs && npm run build   # nginx root 指向 out/
# 浏览器 Disable cache + Ctrl+Shift+R
```

1. 检索 → 自动跳结果页 → **F5 刷新** → 列表仍显示（含文档行内嵌 PDF、行按钮）
2. 聊天页会话中检索 → 结果卡片出现 → **F5 刷新聊天页** → 结果卡片恢复（含"在结果页查看"）
3. 关标签页 → 重开 `test.copiioai.com/app/chat` → 会话恢复 → 结果卡片仍在
4. 同 queryText 检索两次 → 刷新后恢复的是最新一次的结果集
5. 无痕窗口检索 → 刷新 → 无结果（静默降级，不报错）
6. 空页兜底回归：检索后自动跳转（偶发空页场景）→ 不再出现空页
