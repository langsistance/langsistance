# 小程序专利结果列表 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让小程序能查看专利结果的真实详情——列表（标题 + meta）、全字段详情、说明书 PDF、权利要求，并在重开小程序后仍能回看。

**Architecture:** 数据早已随 json 工件到达小程序（`onArtifactsReady` 已收到，只是被渲染层过滤）。本计划把这条数据流接上一个消费方：纯函数层负责解码/取列/裁剪，store 负责内存与持久化，`pages/results` 负责呈现。说明书/权利要求走两个既有的后端接口。

**Tech Stack:** Taro 3 + React（小程序端）、`node --test`（纯函数测试）、FastAPI 既有接口

**Spec:** `docs/superpowers/specs/2026-09-11-weapp-patent-results-design.md`

## Global Constraints

以下每条都来自 spec，**每个任务都隐含包含**：

- **`wx.openDocument` 的 `fileType` 白名单没有 csv / md**；PDF 在白名单内可用
- **小程序没有 `Blob` / `URL.createObjectURL` / `<a download>`**
- **SCSS 的 `px` 会被 Taro 编译成 `rpx`；内联 style 必须写 `rpx`**
- **本仓库组件自 import 样式，页面与全局样式表中没有任何 `@import`**——不要向页面样式表添加组件 SCSS 的 `@import`
- **base64 必须逐 chunk 独立解码再合并字节**，不能先拼字符串再解——256KB 的分片边界会切断 base64 三元组（web 在 `frontend/nextjs/lib/chatSession.js:134-161` 踩过，注释还在）
- **spec/claims 接口失败返回 HTTP 200 + `{success:false}`**，不是 5xx（Cloudflare 会替换源站 5xx 页面，`api_routes/patent_detail.py:774-778`）。**必须按 `success` 判定**，只看状态码会漏
- **存储裁剪值**：每集 40 行 / 摘要 400 字 / 20 集 / index 40 条。小程序总量 10MB、**单键 1MB**（浏览器无单键限制，故比 web 的 50/500/100 更紧）
- **`source` 行级优先于 payload 级**（`frontend/nextjs/lib/results.js:66-70`）
- **spec 用 `patentId || applicationNumber`；claims 用 `applicationNumber || patentId`**（对齐 web `SpecTab.tsx:19` / `ClaimsTab.tsx:21`）
- **服务端跑 Python 3.11**——本计划不改后端
- **验证口径**：`npm run test`（node --test）+ `npm run tsc` + `npm run build:weapp`

---

### Task 1: 纯函数层与 `node --test` 基础设施

**Files:**
- Create: `frontend/weapp/src/utils/results.ts`
- Create: `frontend/weapp/src/utils/results.test.mjs`
- Modify: `frontend/weapp/package.json`

**Interfaces:**
- Consumes: 无（本计划的地基）
- Produces:
  - `ResultColumn { key: string; label: string; role: string }`
  - `ResultsPayload { setId: string; source: string; columns: ResultColumn[]; rows: Array<Record<string,string>> }`
  - `decodeArtifactChunks(chunks: string[]): ResultsPayload | null`
  - `pickColumn(payload: ResultsPayload, role: string): string`
  - `META_ROLES: string[]`
  - `metaLine(payload: ResultsPayload, row: Record<string,string>): string[]`
  - `pruneResults(payload: ResultsPayload): ResultsPayload`
  - `MAX_PERSIST_ROWS = 40`、`MAX_PERSIST_ABSTRACT_CHARS = 400`

**为什么这批函数值得真 TDD**：它们错了**不会报错**，只会安静地显示错数据。§8 的四个断言点全部无 I/O、无平台 API。

- [ ] **Step 1: 加 `test` 脚本**

`frontend/weapp/package.json` 的 `scripts` 加一行（与既有三个同级）：

```json
    "test": "tsc -p tsconfig.test.json && node --test dist-test/**/*.test.mjs"
```

（仓库已有先例：`frontend/nextjs/lib/results.test.mjs` 就是 `node --test` 的 `.mjs` 测试。）

- [ ] **Step 2: 加测试编译配置**

`node --test` 不能直接 import `.ts`，所以测试文件与被测的纯函数先用 `tsc` 编到 `dist-test/`，再让 node 跑编译产物。

创建 `frontend/weapp/tsconfig.test.json`：

```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "noEmit": false,
    "outDir": "dist-test",
    "module": "esnext",
    "target": "es2020",
    "moduleResolution": "bundler"
  },
  "include": ["src/utils/**/*.ts", "src/utils/**/*.mjs"]
}
```

把 `package.json` 的 `test` 脚本改为：

```json
    "test": "tsc -p tsconfig.test.json && node --test dist-test/**/*.test.mjs"
```

把 `dist-test` 加进 `frontend/weapp/.gitignore`（该文件已存在，追加一行）。

- [ ] **Step 3: 写真正的失败测试**

`frontend/weapp/src/utils/results.test.mjs`：

```js
import test from 'node:test'
import assert from 'node:assert/strict'
import {
  decodeArtifactChunks,
  pickColumn,
  metaLine,
  pruneResults,
  MAX_PERSIST_ROWS,
  MAX_PERSIST_ABSTRACT_CHARS,
} from './results.js'

/** 造一份 payload，字段名贴近真实导出（result_export.py 的扁平键） */
function payload(overrides = {}) {
  return {
    setId: 'set-1',
    source: 'uspto',
    columns: [
      { key: 'patentTitle', label: '标题', role: 'title' },
      { key: 'patentNumber', label: '专利号', role: 'patent_id' },
      { key: 'assigneeEntityName', label: '申请人', role: 'assignee' },
      { key: 'abstractText', label: '摘要', role: 'abstract' },
      { key: 'downloadUrl', label: '下载链接', role: 'url' },
    ],
    rows: [{ patentTitle: 'A', patentNumber: 'US1', assigneeEntityName: 'X', abstractText: 'a', downloadUrl: 'u' }],
    ...overrides,
  }
}

const b64 = (s) => Buffer.from(s, 'utf8').toString('base64')

test('decodeArtifactChunks 逐 chunk 独立解码——拼字符串再解会在分片边界解错', () => {
  // 关键：这里的分片边界故意切在一个多字节字符的中间
  const text = JSON.stringify(payload())
  const bytes = Buffer.from(text, 'utf8')
  const mid = 10
  const chunks = [
    bytes.subarray(0, mid).toString('base64'),
    bytes.subarray(mid).toString('base64'),
  ]
  const out = decodeArtifactChunks(chunks)
  assert.equal(out.setId, 'set-1')
  assert.equal(out.rows.length, 1)
})

test('decodeArtifactChunks 载荷非法时返回 null，不抛', () => {
  assert.equal(decodeArtifactChunks([]), null)
  assert.equal(decodeArtifactChunks([b64('not json')]), null)
  assert.equal(decodeArtifactChunks([b64('{"rows":"x"}')]), null) // rows 非数组
})

test('pickColumn 按 role 取值，无匹配返回空串', () => {
  const p = payload()
  assert.equal(pickColumn(p, 'title', p.rows[0]), 'A')
  assert.equal(pickColumn(p, 'patent_id', p.rows[0]), 'US1')
  assert.equal(pickColumn(p, 'nope', p.rows[0]), '')
})

test('metaLine 只收录有值的 role，顺序固定，用 · 连接', () => {
  const p = payload()
  const withAll = {
    ...p.rows[0],
    applicationNumberText: '17638216',
    publicationNumber: 'US20220294065A1',
  }
  p.columns = [
    ...p.columns,
    { key: 'applicationNumberText', label: '申请号', role: 'application_number' },
    { key: 'publicationNumber', label: '公开号', role: 'publication_number' },
  ]
  // patent_id → application_number → publication_number → assignee → publication_date
  assert.deepEqual(metaLine(p, withAll), ['US1', '17638216', 'US20220294065A1', 'X'])
  // 空值不进 meta
  assert.deepEqual(metaLine(p, { ...p.rows[0], assigneeEntityName: '' }), ['US1'])
})

test('pruneResults 裁行数、截摘要、且绝不误删 url 列', () => {
  const p = payload()
  p.rows = Array.from({ length: 100 }, (_, i) => ({
    patentTitle: `T${i}`, patentNumber: `US${i}`,
    assigneeEntityName: 'X', abstractText: 'a'.repeat(2000), downloadUrl: `u${i}`,
  }))
  const out = pruneResults(p)
  assert.equal(out.rows.length, MAX_PERSIST_ROWS)
  assert.equal(out.rows.length, 40)
  assert.equal(out.rows[0].abstractText.length, MAX_PERSIST_ABSTRACT_CHARS)
  assert.equal(out.rows[0].abstractText.length, 400)
  // url 列必须留着——说明书 tab 要用
  const roles = out.columns.map((c) => c.role)
  assert.ok(roles.includes('url'), 'url 列被裁掉了，说明书 tab 会失效')
  assert.ok(roles.includes('abstract'))
  // 前 40 行按原顺序保留
  assert.equal(out.rows[0].patentTitle, 'T0')
  assert.equal(out.rows[39].patentTitle, 'T39')
})

test('pruneResults 不动 setId/source/columns 的非摘要部分', () => {
  const p = payload()
  const out = pruneResults(p)
  assert.equal(out.setId, 'set-1')
  assert.equal(out.source, 'uspto')
  assert.equal(out.columns.length, p.columns.length)
})
```

- [ ] **Step 4: 跑测试确认失败**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run test
```

预期：失败——`src/utils/results.ts` 尚不存在，`tsc -p tsconfig.test.json` 报找不到模块。

- [ ] **Step 5: 实现**

创建 `frontend/weapp/src/utils/results.ts`：

```ts
/**
 * 结果集的纯函数层：解码、按 role 取列、拼 meta、裁剪。
 *
 * 这一层没有 I/O、没有平台 API，所以能真测——而它出的错**不会报错**，
 * 只会安静地显示错数据（少了 url 列就点不开说明书，meta 顺序错了就是
 * 字段对不上号）。故与 store 分开：纯逻辑在此，I/O 在 services/resultsStore.ts。
 */

export interface ResultColumn {
  key: string
  label: string
  role: string
}

export interface ResultsPayload {
  setId: string
  source: string
  columns: ResultColumn[]
  rows: Array<Record<string, string>>
}

/** 每集保留的行数上限。小程序单键 1MB（浏览器无此限制），故比 web 的 50 紧。 */
export const MAX_PERSIST_ROWS = 40
/** 摘要截断。它通常是最长字段。web 是 500。 */
export const MAX_PERSIST_ABSTRACT_CHARS = 400

/**
 * meta 行的 role 与顺序，对齐 web 的 frontend/nextjs/lib/results.js:34。
 * 只收录**有值**的。
 */
export const META_ROLES = [
  'patent_id',
  'application_number',
  'publication_number',
  'assignee',
  'publication_date',
]

/**
 * 逐 chunk 独立解 base64 再合并字节。
 *
 * **不能先拼字符串再解**：后端按 256KB 切片（sse_callback.py:11），分片边界
 * 会切断 base64 三元组，拼起来解就是坏的。web 在 chatSession.js:134-161
 * 踩过这个坑，注释还在。
 *
 * 任何一步失败返回 null（对齐 web 的静默跳过，不打断对话）。
 */
export function decodeArtifactChunks(chunks: string[]): ResultsPayload | null {
  if (!Array.isArray(chunks) || chunks.length === 0) return null
  try {
    // 微信小程序无 Buffer/atob —— 用 base64 字符表手写解码，两处运行时通用
    const BYTES = new Uint8Array(totalBytes(chunks))
    let offset = 0
    for (const chunk of chunks) {
      const bytes = base64ToBytes(chunk)
      if (!bytes) return null
      BYTES.set(bytes, offset)
      offset += bytes.length
    }
    const text = utf8Decode(BYTES)
    const parsed = JSON.parse(text)
    if (!parsed || !Array.isArray(parsed.rows)) return null
    return {
      setId: String(parsed.setId || ''),
      source: String(parsed.source || 'uspto'),
      columns: Array.isArray(parsed.columns) ? parsed.columns : [],
      rows: parsed.rows,
    }
  } catch {
    return null
  }
}

const B64_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

function base64ToBytes(input: string): Uint8Array | null {
  const clean = input.replace(/[\r\n\s]/g, '')
  if (!clean) return null
  let len = clean.length
  while (len % 4 !== 0) len++
  const out = new Uint8Array((len / 4) * 3)
  let o = 0
  for (let i = 0; i < len; i += 4) {
    const c0 = B64_CHARS.indexOf(clean[i] || '=')
    const c1 = B64_CHARS.indexOf(clean[i + 1] || '=')
    const c2 = B64_CHARS.indexOf(clean[i + 2] || '=')
    const c3 = B64_CHARS.indexOf(clean[i + 3] || '=')
    const n = (c0 << 18) | (c1 << 12) | (c2 << 6) | c3
    out[o++] = (n >> 16) & 0xff
    out[o++] = (n >> 8) & 0xff
    out[o++] = n & 0xff
  }
  const pad = (clean.match(/=+$/) || [''])[0].length
  return pad ? out.subarray(0, out.length - pad) : out
}

function totalBytes(chunks: string[]): number {
  let n = 0
  for (const c of chunks) {
    const clean = (c || '').replace(/[\r\n\s]/g, '')
    const pad = (clean.match(/=+$/) || [''])[0].length
    n += (clean.length / 4) * 3 - pad
  }
  return n
}

/** UTF-8 解码。优先用 TextDecoder，缺失时走手写回退（小程序基础库较全，一般走前者）。 */
function utf8Decode(bytes: Uint8Array): string {
  // eslint-disable-next-line no-undef
  if (typeof TextDecoder !== 'undefined') {
    // eslint-disable-next-line no-undef
    return new TextDecoder('utf-8').decode(bytes)
  }
  let out = ''
  let i = 0
  while (i < bytes.length) {
    const b = bytes[i]
    if (b < 0x80) {
      out += String.fromCharCode(b)
      i += 1
    } else if (b < 0xe0) {
      out += String.fromCharCode(((b & 0x1f) << 6) | (bytes[i + 1] & 0x3f))
      i += 2
    } else if (b < 0xf0) {
      out += String.fromCharCode(
        ((b & 0x0f) << 12) | ((bytes[i + 1] & 0x3f) << 6) | (bytes[i + 2] & 0x3f),
      )
      i += 3
    } else {
      const cp =
        ((b & 0x07) << 18) |
        ((bytes[i + 1] & 0x3f) << 12) |
        ((bytes[i + 2] & 0x3f) << 6) |
        (bytes[i + 3] & 0x3f)
      const s = cp - 0x10000
      out += String.fromCharCode(0xd800 + (s >> 10), 0xdc00 + (s & 0x3ff))
      i += 4
    }
  }
  return out
}

/** 按 role 取该行对应列的值；无此 role 或值为空返回 ''。 */
export function pickColumn(
  payload: ResultsPayload,
  role: string,
  row: Record<string, string>,
): string {
  const col = payload.columns.find((c) => c.role === role)
  if (!col) return ''
  return String(row[col.key] ?? '')
}

/** 拼 meta 行：只收录 META_ROLES 中**有值**的项，顺序固定，用 ' · ' 连接。 */
export function metaLine(
  payload: ResultsPayload,
  row: Record<string, string>,
): string[] {
  const out: string[] = []
  for (const role of META_ROLES) {
    const v = pickColumn(payload, role, row)
    if (v) out.push(v)
  }
  return out
}

/**
 * 裁剪成可持久化的副本：行数、摘要长度、以及列。
 *
 * `url` 列**必须保留**——说明书 tab 从它取 PDF 直链，裁掉就点不开了。
 * 其余 role 为 text 的列（后端会塞一个行级 source 列）也保留，否则
 * 行级 source 丢失会导致详情接口传错 source。
 */
export function pruneResults(payload: ResultsPayload): ResultsPayload {
  const abstractCol = payload.columns.find((c) => c.role === 'abstract')
  const columns = payload.columns.filter(
    (c) => c.role !== 'text' || c.key === 'source',
  )
  const rows = payload.rows.slice(0, MAX_PERSIST_ROWS).map((row) => {
    const next: Record<string, string> = { ...row }
    if (abstractCol) {
      const v = next[abstractCol.key]
      if (typeof v === 'string' && v.length > MAX_PERSIST_ABSTRACT_CHARS) {
        next[abstractCol.key] = v.slice(0, MAX_PERSIST_ABSTRACT_CHARS)
      }
    }
    return next
  })
  return {
    setId: payload.setId,
    source: payload.source,
    columns,
    rows,
  }
}
```

- [ ] **Step 6: 跑测试确认通过**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run test
```

预期：全部 PASS（6 个测试）。若 `tsc -p tsconfig.test.json` 因 `allowImportingTsExtensions` 等选项报错，按报错调整 `tsconfig.test.json` 的编译选项，**不要改测试逻辑**。

- [ ] **Step 7: 类型检查**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc
```

预期：exit 0。

- [ ] **Step 8: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add frontend/weapp/src/utils/results.ts \
          frontend/weapp/src/utils/results.test.mjs \
          frontend/weapp/tsconfig.test.json \
          frontend/weapp/package.json && \
  git commit -m "feat(weapp): 结果集纯函数层 + node --test 基础设施"
```

---

### Task 2: 内存 store 与持久化

**Files:**
- Create: `frontend/weapp/src/services/resultsStore.ts`
- Create: `frontend/weapp/src/services/resultsStore.test.mjs`
- Modify: `frontend/weapp/package.json`（test 脚本的 glob 扩到 services）

**Interfaces:**
- Consumes: `ResultsPayload`、`pruneResults`（Task 1）
- Produces:
  - `STORAGE_KEY = 'copiioai_results'`
  - `MAX_RESULT_SETS = 20`、`MAX_INDEX_ENTRIES = 40`
  - `createResultsStore(storage: StorageAdapter)` → `ResultsStore`
  - `StorageAdapter { getSync(key): any; setSync(key, value): void }`
  - `ResultsStore { put(payload): void; get(setId): ResultsPayload | null; load(setId): ResultsPayload | null; persist(payload, meta): void }`
  - `index` 数组的用途**只有淘汰排序**（`savedAt`）。不做按 `queryText` 反查——那是 web 的脆弱做法，spec §5.3 明确改用了消息上的 `set_id`
  - 默认导出 `resultsStore`（用 Taro 的存储实现）

**为什么注入 storage**：直接调 `Taro.setStorageSync` 就没法在 `node --test` 里测。注入之后，用内存假实现就能真测配额与淘汰逻辑。

- [ ] **Step 1: 扩 test 脚本**

`frontend/weapp/package.json`：

```json
    "test": "tsc -p tsconfig.test.json && node --test dist-test/**/*.test.mjs"
```

`tsconfig.test.json` 的 `include` 改为：

```json
  "include": ["src/utils/**/*.ts", "src/utils/**/*.mjs", "src/services/resultsStore.ts", "src/services/resultsStore.test.mjs"]
```

- [ ] **Step 2: 写失败测试**

创建 `frontend/weapp/src/services/resultsStore.test.mjs`：

```js
import test from 'node:test'
import assert from 'node:assert/strict'
import { createResultsStore, STORAGE_KEY, MAX_RESULT_SETS } from './resultsStore.js'

/** 内存假存储，可模拟写满 */
function fakeStorage({ failOnWrite = false } = {}) {
  const map = new Map()
  return {
    getSync: (k) => (map.has(k) ? map.get(k) : null),
    setSync: (k, v) => {
      if (failOnWrite) throw new Error('quota exceeded')
      map.set(k, v)
    },
    _dump: () => Object.fromEntries(map),
  }
}

function payload(setId, rows = 3) {
  return {
    setId,
    source: 'uspto',
    columns: [
      { key: 'patentTitle', label: '标题', role: 'title' },
      { key: 'downloadUrl', label: '链接', role: 'url' },
    ],
    rows: Array.from({ length: rows }, (_, i) => ({ patentTitle: `T${i}`, downloadUrl: `u${i}` })),
  }
}

test('put/get 走内存——刚收到的结果集立即可取', () => {
  const store = createResultsStore(fakeStorage())
  store.put(payload('a'))
  assert.equal(store.get('a').rows.length, 3)
  assert.equal(store.get('nope'), null)
})

test('persist 落盘后才可跨实例取回', () => {
  const storage = fakeStorage()
  createResultsStore(storage).persist(payload('a'), { sessionId: 's1', queryText: 'q' })
  // 新实例（模拟重开小程序）只能从 storage 读
  const reopened = createResultsStore(storage)
  assert.equal(reopened.get('a'), null, '新实例的内存应为空')
  assert.equal(reopened.load('a').rows.length, 3)
})

test('超过 MAX_RESULT_SETS 时丢最旧的（按 savedAt，不是对象键序）', () => {
  const storage = fakeStorage()
  const store = createResultsStore(storage)
  for (let i = 0; i < MAX_RESULT_SETS + 3; i++) {
    store.persist(payload(`set-${i}`), { sessionId: 's', queryText: `q${i}`, savedAt: 1000 + i })
  }
  const raw = storage.getSync(STORAGE_KEY)
  const ids = Object.keys(raw.sets)
  assert.equal(ids.length, MAX_RESULT_SETS)
  assert.ok(!ids.includes('set-0'), '最旧的没被丢掉')
  assert.ok(!ids.includes('set-2'))
  assert.ok(ids.includes('set-3'), '边界丢多了')
  assert.ok(ids.includes(`set-${MAX_RESULT_SETS + 2}`))
})

test('写盘失败静默放弃，不抛——配额问题不能打断对话', () => {
  const store = createResultsStore(fakeStorage({ failOnWrite: true }))
  assert.doesNotThrow(() => store.persist(payload('a'), { sessionId: 's', queryText: 'q' }))
  // 内存仍然可用
  assert.equal(store.get('a').rows.length, 3)
})
```

- [ ] **Step 3: 跑测试确认失败**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run test
```

预期：失败——`resultsStore.ts` 不存在。

- [ ] **Step 4: 实现**

创建 `frontend/weapp/src/services/resultsStore.ts`：

```ts
import Taro from '@tarojs/taro'
import { ResultsPayload, pruneResults } from '../utils/results'

/**
 * 结果集的内存态 + 持久化。
 *
 * 纯逻辑（解码/取列/裁剪）在 utils/results.ts，这里只做状态与 I/O。
 * storage 走注入而非直接调 Taro —— 否则配额与淘汰逻辑没法在 node 里测。
 */

/** 与 web 同键名（frontend/nextjs/lib/resultsStore.js:11），便于将来对齐。 */
export const STORAGE_KEY = 'copiioai_results'

/** 保留的集数上限。小程序总量 10MB，20 集 × 约 40KB ≈ 800KB，留足余量。web 是 100。 */
export const MAX_RESULT_SETS = 20
/** index 条目上限。与集数同量级即可。web 是 200。 */
export const MAX_INDEX_ENTRIES = 40

export interface StorageAdapter {
  getSync(key: string): any
  setSync(key: string, value: any): void
}

export interface IndexEntry {
  setId: string
  sessionId: string
  queryText: string
  savedAt: number
}

interface StoredShape {
  sets: Record<string, ResultsPayload>
  index: IndexEntry[]
}

function emptyShape(): StoredShape {
  return { sets: {}, index: [] }
}

export interface ResultsStore {
  put(payload: ResultsPayload): void
  get(setId: string): ResultsPayload | null
  load(setId: string): ResultsPayload | null
  persist(payload: ResultsPayload, meta: { sessionId: string; queryText: string; savedAt?: number }): void
}

export function createResultsStore(storage: StorageAdapter): ResultsStore {
  // 当前会话的内存态。全量——持久化那份是裁过的。
  const mem = new Map<string, ResultsPayload>()

  function read(): StoredShape {
    try {
      const raw = storage.getSync(STORAGE_KEY)
      if (!raw || typeof raw !== 'object') return emptyShape()
      return {
        sets: raw.sets && typeof raw.sets === 'object' ? raw.sets : {},
        index: Array.isArray(raw.index) ? raw.index : [],
      }
    } catch {
      return emptyShape()
    }
  }

  return {
    put(payload) {
      mem.set(payload.setId, payload)
    },

    get(setId) {
      return mem.get(setId) || null
    },

    load(setId) {
      return read().sets[setId] || null
    },

    persist(payload, meta) {
      // 内存先写：即使落盘失败，当前会话也要能用
      mem.set(payload.setId, payload)
      const savedAt = meta.savedAt ?? Date.now()
      try {
        const shape = read()
        shape.sets[payload.setId] = pruneResults(payload)
        shape.index = [
          { setId: payload.setId, sessionId: meta.sessionId, queryText: meta.queryText, savedAt },
          ...shape.index.filter((e) => e.setId !== payload.setId),
        ]
        // 淘汰**按 savedAt**，不是对象键序——web 的 dropOldestSet 用的是
        // Object.keys 顺序，与它自己的 index 插入序并不一致，那个缺陷不照搬。
        const byOldest = Object.entries(shape.sets).sort(
          (a, b) => (findSavedAt(shape, a[0]) - findSavedAt(shape, b[0])),
        )
        for (const [id] of byOldest.slice(0, Math.max(0, byOldest.length - MAX_RESULT_SETS))) {
          delete shape.sets[id]
        }
        shape.index = shape.index
          .sort((a, b) => b.savedAt - a.savedAt)
          .slice(0, MAX_INDEX_ENTRIES)
        storage.setSync(STORAGE_KEY, shape)
      } catch {
        // 写满/配额超限：静默放弃。结果集是锦上添花，不能打断对话。
      }
    },
  }
}

function findSavedAt(shape: StoredShape, setId: string): number {
  const entry = shape.index.find((e) => e.setId === setId)
  return entry ? entry.savedAt : 0
}

/** 默认实例：接到 Taro 的同步存储。 */
export const resultsStore = createResultsStore({
  getSync: (key) => Taro.getStorageSync(key),
  setSync: (key, value) => Taro.setStorageSync(key, value),
})
```

- [ ] **Step 5: 跑测试确认通过**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run test
```

预期：全部 PASS（Task 1 的 6 个 + 本任务 5 个 = 11 个）。

- [ ] **Step 6: 类型检查 + 构建**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc && npm run build:weapp
```

预期：`tsc` exit 0；`Compiled successfully`。

- [ ] **Step 7: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add frontend/weapp/src/services/resultsStore.ts \
          frontend/weapp/src/services/resultsStore.test.mjs \
          frontend/weapp/tsconfig.test.json \
          frontend/weapp/package.json && \
  git commit -m "feat(weapp): 结果集 store——内存态 + 持久化 + 按 savedAt 淘汰"
```

---

### Task 3: 详情接口服务（spec / claims）

**Files:**
- Create: `frontend/weapp/src/services/patentDetail.ts`
- Create: `frontend/weapp/src/services/patentDetail.test.mjs`
- Modify: `frontend/weapp/tsconfig.test.json`（include 加本文件）

**Interfaces:**
- Consumes: `request`（`services/api.ts`）
- Produces:
  - `normalizeSpec(body): { ok: boolean; pdfUrl: string; message: string }`
  - `normalizeClaims(body): { ok: boolean; claims: ClaimItem[]; pdfUrl: string; message: string }`
  - `ClaimItem { number: number; text: string; independent: boolean }`
  - `fetchSpec(source, patentId, applicationNumber?): Promise<{ok,pdfUrl,message}>`
  - `fetchClaims(source, patentId, applicationNumber?): Promise<{ok,claims,pdfUrl,message}>`

**关键约束**：接口失败返回 **HTTP 200 + `{success:false}`**（`patent_detail.py:774-778`）。按状态码判会漏掉全部业务失败。`normalize*` 是纯函数，所以这部分能真测。

- [ ] **Step 1: 写失败测试**

创建 `frontend/weapp/src/services/patentDetail.test.mjs`：

```js
import test from 'node:test'
import assert from 'node:assert/strict'
import { normalizeSpec, normalizeClaims, specTarget, claimsTarget } from './patentDetail.js'

test('normalizeSpec：success:false 判为失败——HTTP 200 也可能是业务失败', () => {
  const out = normalizeSpec({ success: false, message: '未找到说明书' })
  assert.equal(out.ok, false)
  assert.equal(out.message, '未找到说明书')
  assert.equal(out.pdfUrl, '')
})

test('normalizeSpec：success:true 取 pdf_url', () => {
  const out = normalizeSpec({ success: true, pdf_url: 'https://x/a.pdf' })
  assert.equal(out.ok, true)
  assert.equal(out.pdfUrl, 'https://x/a.pdf')
})

test('normalizeSpec：success 为真但无 pdf_url，判为失败而不是给空链接', () => {
  const out = normalizeSpec({ success: true })
  assert.equal(out.ok, false)
})

test('normalizeClaims：结构化权利要求归一，只留用到的三个字段', () => {
  const out = normalizeClaims({
    success: true,
    claims: [
      { number: 1, text: '一种…', status: 'x', independent: true },
      { number: 2, text: '根据权利要求1…', independent: false },
    ],
  })
  assert.equal(out.ok, true)
  assert.deepEqual(out.claims, [
    { number: 1, text: '一种…', independent: true },
    { number: 2, text: '根据权利要求1…', independent: false },
  ])
})

test('normalizeClaims：无结构化但有 pdf_url → ok 且 claims 为空，调用方回退 PDF', () => {
  const out = normalizeClaims({ success: true, pdf_url: 'https://x/c.pdf' })
  assert.equal(out.ok, true)
  assert.equal(out.claims.length, 0)
  assert.equal(out.pdfUrl, 'https://x/c.pdf')
})

test('normalizeClaims：既无 claims 也无 pdf_url → 失败', () => {
  assert.equal(normalizeClaims({ success: true }).ok, false)
})

test('spec 用 patentId 优先，claims 用 applicationNumber 优先（对齐 web）', () => {
  assert.equal(specTarget('US1', '17638216'), 'US1')
  assert.equal(specTarget('', '17638216'), '17638216')
  assert.equal(specTarget('', ''), '')
  assert.equal(claimsTarget('US1', '17638216'), '17638216')
  assert.equal(claimsTarget('US1', ''), 'US1')
  assert.equal(claimsTarget('', ''), '')
})
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run test
```

预期：失败——`patentDetail.ts` 不存在。

- [ ] **Step 3: 实现**

创建 `frontend/weapp/src/services/patentDetail.ts`：

```ts
import { request } from './api'

/**
 * 专利详情接口：说明书（spec）与权利要求（claims）。
 *
 * ⚠️ 这两条接口**失败时返回 HTTP 200 + {success:false}**，不是 5xx——
 * Cloudflare 会替换源站的 5xx 页面并导致 CORS 失败（api_routes/patent_detail.py:774-778）。
 * 所以判定必须看 `success` 字段，只看状态码会漏掉全部业务失败。
 *
 * 归一层是纯函数，故可测；网络部分只做参数拼装与转发。
 */

export interface ClaimItem {
  number: number
  text: string
  independent: boolean
}

export interface SpecResult {
  ok: boolean
  pdfUrl: string
  message: string
}

export interface ClaimsResult {
  ok: boolean
  claims: ClaimItem[]
  pdfUrl: string
  message: string
}

/** spec 用 patentId 优先，回退 applicationNumber（对齐 web SpecTab.tsx:19）。 */
export function specTarget(patentId: string, applicationNumber: string): string {
  return patentId || applicationNumber || ''
}

/** claims 用 applicationNumber 优先，回退 patentId（对齐 web ClaimsTab.tsx:21）。 */
export function claimsTarget(patentId: string, applicationNumber: string): string {
  return applicationNumber || patentId || ''
}

export function normalizeSpec(body: any): SpecResult {
  const message = String((body && body.message) || '')
  if (!body || body.success !== true) {
    return { ok: false, pdfUrl: '', message: message || '未找到说明书' }
  }
  const pdfUrl = String(body.pdf_url || '')
  if (!pdfUrl) {
    // success 但没给链接：当作失败，不要把空串交给下载层
    return { ok: false, pdfUrl: '', message: message || '未找到说明书' }
  }
  return { ok: true, pdfUrl, message }
}

export function normalizeClaims(body: any): ClaimsResult {
  const message = String((body && body.message) || '')
  const pdfUrl = String((body && body.pdf_url) || '')
  if (!body || body.success !== true) {
    return { ok: false, claims: [], pdfUrl: '', message: message || '未找到权利要求' }
  }
  const raw = Array.isArray(body.claims) ? body.claims : []
  // 只留渲染要用的三个字段。status 的取值语义未核实，不猜、不透传。
  const claims: ClaimItem[] = raw.map((c: any) => ({
    number: Number(c && c.number) || 0,
    text: String((c && c.text) || ''),
    independent: Boolean(c && c.independent),
  }))
  if (claims.length === 0 && !pdfUrl) {
    return { ok: false, claims: [], pdfUrl: '', message: message || '未找到权利要求' }
  }
  return { ok: true, claims, pdfUrl, message }
}

export async function fetchSpec(
  source: string,
  patentId: string,
  applicationNumber = '',
): Promise<SpecResult> {
  const target = specTarget(patentId, applicationNumber)
  if (!target) return { ok: false, pdfUrl: '', message: '该条缺少可查询的专利号' }
  const body = await request(
    `/patent/${encodeURIComponent(source)}/${encodeURIComponent(target)}/spec`,
  )
  return normalizeSpec(body)
}

export async function fetchClaims(
  source: string,
  patentId: string,
  applicationNumber = '',
): Promise<ClaimsResult> {
  const target = claimsTarget(patentId, applicationNumber)
  if (!target) return { ok: false, claims: [], pdfUrl: '', message: '该条缺少可查询的专利号' }
  const body = await request(
    `/patent/${encodeURIComponent(source)}/${encodeURIComponent(target)}/claims`,
  )
  return normalizeClaims(body)
}
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run test
```

预期：全部 PASS（11 + 7 = 18 个）。

- [ ] **Step 5: 类型检查 + 构建**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc && npm run build:weapp
```

- [ ] **Step 6: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add frontend/weapp/src/services/patentDetail.ts \
          frontend/weapp/src/services/patentDetail.test.mjs \
          frontend/weapp/tsconfig.test.json && \
  git commit -m "feat(weapp): 专利详情接口服务——按 success 判定，不只看状态码"
```

---

### Task 4: 对话页接住 json 工件并给出结果入口

**Files:**
- Modify: `frontend/weapp/src/pages/chat/index.tsx`
- Modify: `frontend/weapp/src/pages/chat/index.scss`

**Interfaces:**
- Consumes: `decodeArtifactChunks`（Task 1）、`resultsStore`（Task 2）
- Produces: `MsgView.resultSet?: { setId: string; rowCount: number }`

**三条硬要求**：

1. **json 工件解码入库后立刻从 `artifacts` 移除**——它的 base64 是 multi-MB，而 store 里已有解码后的同一份数据。不移除就是长期占着内存。
2. **入口按 `m.resultSet` 渲染，不是按 `artifacts` 里有没有 json**（json 已经被移除了）。
3. **保存消息时带上 `set_id`**，历史会话才能找回结果。后端对消息数组是逐字透传的（`api_routes/session.py:202-224`），加字段不需要改后端。

- [ ] **Step 1: 扩展 MsgView**

`frontend/weapp/src/pages/chat/index.tsx` 的 `MsgView` 接口追加：

```tsx
  /** 本轮结果集的引用。json 工件的 base64 解码入库后就不再留在消息里。 */
  resultSet?: { setId: string; rowCount: number }
```

- [ ] **Step 2: `onArtifactsReady` 分流**

把现有的 `onArtifactsReady` 回调替换为：

```tsx
  const onArtifactsReady = useCallback((items: CompletedArtifact[]) => {
    if (items.length === 0) return
    // 先分流：json 是结果集的载体，走 store；其余是下载工件，留在消息上
    let resultSet: { setId: string; rowCount: number } | null = null
    const downloadable: CompletedArtifact[] = []
    for (const it of items) {
      if (it.format === 'json') {
        const decoded = decodeArtifactChunks(it.chunks)
        if (decoded) {
          decoded.setId = it.artifactId
          resultsStore.put(decoded)
          resultSet = { setId: it.artifactId, rowCount: decoded.rows.length }
        }
        // 解码失败的 json 直接丢——留着只占内存，没有任何消费方
      } else {
        downloadable.push(it)
      }
    }
    setMsgs((prev) => {
      const next = prev.slice()
      const last = next[next.length - 1]
      if (last && last.role === 'assistant') {
        next[next.length - 1] = {
          ...last,
          artifacts: [...(last.artifacts || []), ...downloadable],
          ...(resultSet ? { resultSet } : {}),
        }
      }
      return next
    })
  }, [])
```

import 区加：

```tsx
import { decodeArtifactChunks } from '../../utils/results'
import { resultsStore } from '../../services/resultsStore'
```

- [ ] **Step 3: 渲染入口**

在助手消息的工件行**之前**插入（`chat-msg-artifacts` 那个 `<View>` 之前）：

```tsx
              {m.role === 'assistant' && m.resultSet ? (
                <View
                  className='chat-result-entry'
                  onClick={() =>
                    Taro.navigateTo({
                      url: `/pages/results/index?set=${encodeURIComponent(m.resultSet!.setId)}`,
                    })
                  }
                >
                  <Text className='chat-result-entry-label'>
                    查看全部 {m.resultSet.rowCount} 项结果
                  </Text>
                  <Text className='chat-result-entry-arrow'>›</Text>
                </View>
              ) : null}
```

`frontend/weapp/src/pages/chat/index.scss` 追加（**不要往这个文件里加 `@import`**）：

```scss
/* ── 结果集入口 ── */
.chat-result-entry {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 12px;
  padding: 20px 24px;
  border-radius: 12px;
  background: var(--c-primary-soft);
  border: 1px solid rgba(16, 163, 127, 0.25);

  &:active {
    background: #d8ece6;
  }
}

.chat-result-entry-label {
  font-size: var(--fs-meta);
  font-weight: 600;
  color: var(--c-primary-deep);
}

.chat-result-entry-arrow {
  font-size: 32px;
  line-height: 1;
  color: var(--c-primary-deep);
}
```

- [ ] **Step 4: 保存时带 `set_id`**

把普通问答分支里构造 `assistantMsg` 的那段：

```tsx
        const assistantMsg: ChatMsg = {
          role: 'assistant',
          content: assistantRef.current,
        }
        await saveMessages(sid, [...history, userMsg, assistantMsg])
```

替换为：

```tsx
        // 带上 resultSetId：后端对消息数组是逐字透传（session.py:202-224），
        // 加字段不需要改后端。历史会话靠它找回本地结果集。
        const lastResultSet = lastAssistantResultSet()
        const assistantMsg: ChatMsg = {
          role: 'assistant',
          content: assistantRef.current,
          ...(lastResultSet ? { set_id: lastResultSet.setId } : {}),
        }
        await saveMessages(sid, [...history, userMsg, assistantMsg])
        // 结果集落盘（裁剪版），供重开小程序后回看
        if (lastResultSet) {
          resultsStore.persist(resultSetPayload(lastResultSet.setId), {
            sessionId: sid,
            queryText: text,
          })
        }
```

并在 `copyPatent` 之前加两个小工具：

```tsx
  /** 最新一条助手消息上的结果集引用（落库与持久化都要用）。 */
  function lastAssistantResultSet(): { setId: string; rowCount: number } | null {
    for (let i = msgsRef.current.length - 1; i >= 0; i--) {
      const m = msgsRef.current[i]
      if (m.role === 'assistant' && m.resultSet) return m.resultSet
    }
    return null
  }

  /** 从 store 取回刚入库的完整载荷；还没入库就返回 null。 */
  function resultSetPayload(setId: string): ResultsPayload {
    const found = resultsStore.get(setId)
    if (found) return found
    return { setId, source: 'uspto', columns: [], rows: [] }
  }
```

import 区补：

```tsx
import { ResultsPayload } from '../../utils/results'
```

- [ ] **Step 5: 类型检查 + 构建**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc && npm run build:weapp
```

预期：`tsc` exit 0；`Compiled successfully`。

- [ ] **Step 6: 人工验证（开发者工具）**

**完全重启**开发者工具（`compileHotReLoad` 已关）。发一个会产生结果表的提问（如「查一下可折叠桌子相关的专利」）。

预期：回答下方出现「查看全部 N 项结果」，N 与真实行数一致；**没有** json 工件的回答只显示原有的专利号 chip。此时点击入口会因 `pages/results` 还不存在而失败——Task 5 建它。

- [ ] **Step 7: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add frontend/weapp/src/pages/chat/index.tsx \
          frontend/weapp/src/pages/chat/index.scss && \
  git commit -m "feat(weapp): 接住 json 工件并给出结果入口——解码入库后从 artifacts 移除省内存"
```

---

### Task 5: 结果列表页

**Files:**
- Create: `frontend/weapp/src/pages/results/index.tsx`
- Create: `frontend/weapp/src/pages/results/index.config.ts`
- Create: `frontend/weapp/src/pages/results/index.scss`
- Modify: `frontend/weapp/src/app.config.ts`

**Interfaces:**
- Consumes: `resultsStore`（Task 2）、`metaLine` / `pickColumn` / `ResultsPayload`（Task 1）
- Produces: 路由 `/pages/results/index?set=<setId>`

**用系统导航栏**（`navigationBarTitleText: '检索结果'`）——白拿一个原生返回键。对话页自绘 NavBar 是因为 ☰ 要占左上角，这里不需要。

- [ ] **Step 1: 注册路由**

`frontend/weapp/src/app.config.ts` 的 `pages` 追加第三项（`pages/privacy/index` 之后）：

```ts
    'pages/results/index', // 专利结果列表（结果面板）
```

- [ ] **Step 2: 页面配置**

创建 `frontend/weapp/src/pages/results/index.config.ts`：

```ts
export default {
  navigationBarTitleText: '检索结果',
}
```

- [ ] **Step 3: 页面实现**

创建 `frontend/weapp/src/pages/results/index.tsx`：

```tsx
import { useMemo, useState } from 'react'
import { ScrollView, Text, View } from '@tarojs/components'
import { useRouter } from '@tarojs/taro'
import { resultsStore } from '../../services/resultsStore'
import { ResultsPayload, metaLine, pickColumn } from '../../utils/results'
import ResultDetail from '../../components/ResultDetail'
import './index.scss'

/**
 * 专利结果列表。数据来自 resultsStore：
 * 本会话的结果在内存里（全量），重开小程序后从 storage 读回（裁剪版 ≤40 行）。
 */
export default function ResultsPage() {
  const router = useRouter()
  const setId = decodeURIComponent(String(router.params.set || ''))

  // 先查内存（当前会话全量），再回落到 storage（历史裁剪版）
  const payload = useMemo<ResultsPayload | null>(
    () => resultsStore.get(setId) || resultsStore.load(setId),
    [setId],
  )

  const [activeIndex, setActiveIndex] = useState(-1)

  if (!payload || payload.rows.length === 0) {
    return (
      <View className='results-empty'>
        <Text className='results-empty-title'>结果已不可用</Text>
        <Text className='results-empty-hint'>
          本机没有这份结果的缓存（可能换了设备或清理过缓存）。请重新发起检索。
        </Text>
      </View>
    )
  }

  const row = activeIndex >= 0 ? payload.rows[activeIndex] : null

  return (
    <View className='results'>
      <ScrollView className='results-scroll' scrollY>
        <View className='results-count'>共 {payload.rows.length} 项</View>
        {payload.rows.map((r, i) => {
          const title = pickColumn(payload, 'title', r) || '—'
          const meta = metaLine(payload, r)
          return (
            <View
              key={i}
              className='results-row'
              onClick={() => setActiveIndex(i)}
            >
              <View className='results-row-main'>
                <Text className='results-row-title'>{title}</Text>
                {meta.length > 0 ? (
                  <Text className='results-row-meta'>{meta.join(' · ')}</Text>
                ) : null}
              </View>
              <Text className='results-row-arrow'>›</Text>
            </View>
          )
        })}
      </ScrollView>

      {row ? (
        <ResultDetail
          payload={payload}
          row={row}
          visible
          onClose={() => setActiveIndex(-1)}
        />
      ) : null}
    </View>
  )
}
```

创建 `frontend/weapp/src/pages/results/index.scss`：

```scss
.results {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: var(--c-bg);
}

.results-scroll {
  flex: 1;
  min-height: 0;
}

.results-count {
  padding: 20px 28px 8px;
  font-size: var(--fs-meta);
  color: var(--c-text-muted);
}

.results-row {
  display: flex;
  align-items: center;
  margin: 12px 24px 0;
  padding: 24px 26px;
  background: var(--c-surface);
  border-radius: 16px;
  border: 1px solid var(--c-border);

  &:active {
    background: var(--c-primary-soft);
  }
}

.results-row-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.results-row-title {
  font-size: var(--fs-msg);
  font-weight: 600;
  line-height: 1.5;
  color: var(--c-text);
}

.results-row-meta {
  margin-top: 10px;
  font-size: var(--fs-meta);
  line-height: 1.5;
  color: var(--c-text-muted);
}

.results-row-arrow {
  flex-shrink: 0;
  margin-left: 16px;
  font-size: 36px;
  line-height: 1;
  color: var(--c-text-muted);
}

/* ── 空态 ── */
.results-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 200px 60px 0;
}

.results-empty-title {
  font-size: var(--fs-msg);
  font-weight: 600;
  color: var(--c-text);
}

.results-empty-hint {
  margin-top: 20px;
  font-size: var(--fs-meta);
  line-height: 1.7;
  color: var(--c-text-muted);
  text-align: center;
}
```

**注意**：`ResultDetail` 在 Task 6 创建。本任务先建一个**最小占位**让页面能编译：

创建 `frontend/weapp/src/components/ResultDetail/index.tsx`：

```tsx
import { Text, View } from '@tarojs/components'
import { ResultsPayload } from '../../utils/results'
import './index.scss'

type Props = {
  payload: ResultsPayload
  row: Record<string, string>
  visible: boolean
  onClose: () => void
}

/** Task 6 会把它换成真正的三 tab 详情。 */
export default function ResultDetail({ visible, onClose }: Props) {
  if (!visible) return null
  return (
    <View className='rd' onClick={onClose}>
      <Text className='rd-placeholder'>详情面板（Task 6 实现）</Text>
    </View>
  )
}
```

创建 `frontend/weapp/src/components/ResultDetail/index.scss`：

```scss
.rd {
  position: fixed;
  inset: 0;
  z-index: 500;
  background: var(--c-surface);
  padding: 60px 40px;
  box-sizing: border-box;
}

.rd-placeholder {
  font-size: var(--fs-body);
  color: var(--c-text-muted);
}
```

- [ ] **Step 4: 类型检查 + 构建**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc && npm run build:weapp
```

预期：`tsc` exit 0；`Compiled successfully`。用 `grep -o "results" dist/app.json` 确认路由进了产物。

- [ ] **Step 5: 人工验证**

**完全重启**开发者工具。发一个会产生结果表的提问 → 点「查看全部 N 项结果」→ 列表出现，每行有标题与 meta（专利号 · 申请号 · …），点任一行弹出占位面板。

- [ ] **Step 6: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add frontend/weapp/src/pages/results \
          frontend/weapp/src/components/ResultDetail \
          frontend/weapp/src/app.config.ts && \
  git commit -m "feat(weapp): 结果列表页——标题 + meta 行，系统导航栏白拿原生返回"
```

---

### Task 6: 详情面板——三个 tab

**Files:**
- Modify: `frontend/weapp/src/components/ResultDetail/index.tsx`
- Modify: `frontend/weapp/src/components/ResultDetail/index.scss`

**Interfaces:**
- Consumes: `ResultsPayload` / `pickColumn`（Task 1）、`fetchSpec` / `fetchClaims`（Task 3）、`downloadReport` / `openOrShareFile`（`services/download.ts`，M3 已有）
- Produces: 无新导出

**三个 tab 的数据来源**：
- **详情**：全在已收到的 payload 里，**零新增请求**
- **说明书**：`fetchSpec` → `{pdfUrl}` → 下载后打开
- **权利要求**：`fetchClaims` → 结构化渲染；无结构化回退 PDF

- [ ] **Step 1: 实现**

把 `frontend/weapp/src/components/ResultDetail/index.tsx` 整体替换为：

```tsx
import { useState } from 'react'
import { ScrollView, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import { ResultsPayload, pickColumn } from '../../utils/results'
import { ClaimItem, fetchClaims, fetchSpec } from '../../services/patentDetail'
import { openOrShareFile } from '../../services/download'
import { errorText } from '../../services/api'
import './index.scss'

type Tab = 'details' | 'spec' | 'claims'

type Props = {
  payload: ResultsPayload
  row: Record<string, string>
  visible: boolean
  onClose: () => void
}

export default function ResultDetail({ payload, row, visible, onClose }: Props) {
  const [tab, setTab] = useState<Tab>('details')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [claims, setClaims] = useState<ClaimItem[] | null>(null)
  const [claimsPdf, setClaimsPdf] = useState('')

  if (!visible) return null

  const patentId = pickColumn(payload, 'patent_id', row)
  const appNumber = pickColumn(payload, 'application_number', row)
  // 行级 source 优先于 payload 级（对齐 web results.js:66-70）
  const rowSource = String(row.source || '') || payload.source
  const title = pickColumn(payload, 'title', row) || '—'
  const canQuery = Boolean(patentId || appNumber)

  // 详情的字段表：全部**非空**字段，标签用列的中文 label
  const fields = payload.columns
    .map((c) => ({ label: c.label, value: String(row[c.key] ?? '') }))
    .filter((f) => f.value)

  /**
   * 下载 PDF 到临时文件后打开。
   * 不自己调 openDocument——复用 M3 的 openOrShareFile，它已经带了
   * 「打不开就回退到 shareFileMessage 转发」的分支（部分机型对 office/PDF
   * 支持不全）。
   */
  async function openPdf(url: string) {
    const res = await Taro.downloadFile({ url })
    if (res.statusCode >= 400) throw new Error('文件下载失败')
    await openOrShareFile(res.tempFilePath, 'pdf')
  }

  async function loadSpec() {
    if (busy) return
    setBusy(true)
    setError('')
    try {
      const r = await fetchSpec(rowSource, patentId, appNumber)
      if (!r.ok) {
        setError(r.message)
        return
      }
      await openPdf(r.pdfUrl)
    } catch (err) {
      setError(errorText(err, '说明书加载失败'))
    } finally {
      setBusy(false)
    }
  }

  async function loadClaims() {
    if (busy) return
    setBusy(true)
    setError('')
    try {
      const r = await fetchClaims(rowSource, patentId, appNumber)
      if (!r.ok) {
        setError(r.message)
        return
      }
      setClaims(r.claims)
      setClaimsPdf(r.pdfUrl)
      // 后端没给结构化权利要求 → 直接开 PDF
      if (r.claims.length === 0 && r.pdfUrl) {
        await openPdf(r.pdfUrl)
      }
    } catch (err) {
      setError(errorText(err, '权利要求加载失败'))
    } finally {
      setBusy(false)
    }
  }

  function switchTab(next: Tab) {
    setTab(next)
    setError('')
    if (next === 'spec' && claims === null) loadSpec()
    if (next === 'claims' && claims === null) loadClaims()
  }

  return (
    <View className='rd'>
      <View className='rd-head'>
        <View className='rd-back' onClick={onClose}>
          <Text className='rd-back-icon'>‹</Text>
        </View>
        <View className='rd-tabs'>
          {(['details', 'spec', 'claims'] as Tab[]).map((key) => (
            <View
              key={key}
              className={`rd-tab${tab === key ? ' rd-tab-active' : ''}`}
              onClick={() => switchTab(key)}
            >
              <Text>
                {key === 'details' ? '详情' : key === 'spec' ? '说明书' : '权利要求'}
              </Text>
            </View>
          ))}
        </View>
      </View>

      {!canQuery && tab !== 'details' ? (
        <View className='rd-note'>
          <Text>该条缺少可查询的专利号</Text>
        </View>
      ) : null}

      {busy ? (
        <View className='rd-note'>
          <Text>加载中…</Text>
        </View>
      ) : null}

      {error ? (
        <View className='rd-error'>
          <Text>{error}</Text>
        </View>
      ) : null}

      <ScrollView className='rd-body' scrollY>
        {tab === 'details' ? (
          <View className='rd-fields'>
            <Text className='rd-title'>{title}</Text>
            {fields.map((f) => (
              <View key={f.label} className='rd-field'>
                <Text className='rd-field-label'>{f.label}</Text>
                <Text className='rd-field-value'>{f.value}</Text>
              </View>
            ))}
          </View>
        ) : null}

        {tab === 'claims' && claims && claims.length > 0 ? (
          <View className='rd-claims'>
            {claims.map((c) => (
              <View key={c.number} className='rd-claim'>
                <Text className='rd-claim-head'>
                  第 {c.number} 项 · {c.independent ? '独立' : '从属'}
                </Text>
                <Text className='rd-claim-text'>{c.text}</Text>
              </View>
            ))}
          </View>
        ) : null}

        {tab === 'claims' && claims && claims.length === 0 && claimsPdf ? (
          <View className='rd-note'>
            <Text>该专利无结构化权利要求，已打开 PDF</Text>
          </View>
        ) : null}
      </ScrollView>
    </View>
  )
}
```

把 `frontend/weapp/src/components/ResultDetail/index.scss` 整体替换为：

```scss
/* 页内覆盖层：盖住列表页，不跳新页（返回逻辑简单，切 tab 不丢状态） */
.rd {
  position: fixed;
  inset: 0;
  z-index: 500;
  background: var(--c-surface);
  display: flex;
  flex-direction: column;
}

.rd-head {
  display: flex;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid var(--c-border);
}

.rd-back {
  flex-shrink: 0;
  width: 64px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.rd-back-icon {
  font-size: 44px;
  line-height: 1;
  color: var(--c-text);
}

.rd-tabs {
  flex: 1;
  display: flex;
  gap: 12px;
  margin-left: 12px;
}

.rd-tab {
  padding: 10px 24px;
  border-radius: 999px;
  background: var(--c-bg);
  font-size: var(--fs-meta);
  color: var(--c-text-muted);
}

.rd-tab-active {
  background: var(--c-primary-soft);
  color: var(--c-primary-deep);
  font-weight: 600;
}

.rd-body {
  flex: 1;
  min-height: 0;
}

.rd-fields {
  padding: 28px 32px 80px;
  display: flex;
  flex-direction: column;
}

.rd-title {
  font-size: var(--fs-msg);
  font-weight: 600;
  line-height: 1.5;
  color: var(--c-text);
}

.rd-field {
  display: flex;
  margin-top: 24px;
}

.rd-field-label {
  flex-shrink: 0;
  width: 160px;
  font-size: var(--fs-meta);
  color: var(--c-text-muted);
}

.rd-field-value {
  flex: 1;
  min-width: 0;
  font-size: var(--fs-meta);
  line-height: 1.7;
  color: var(--c-text);
  word-break: break-word;
}

.rd-claims {
  padding: 28px 32px 80px;
}

.rd-claim {
  margin-bottom: 32px;
}

.rd-claim-head {
  display: block;
  font-size: var(--fs-meta);
  font-weight: 600;
  color: var(--c-primary-deep);
}

.rd-claim-text {
  display: block;
  margin-top: 12px;
  font-size: var(--fs-meta);
  line-height: 1.75;
  color: var(--c-text);
}

.rd-note {
  padding: 24px 32px;
  font-size: var(--fs-meta);
  color: var(--c-text-muted);
}

.rd-error {
  padding: 24px 32px;
  font-size: var(--fs-meta);
  color: var(--c-danger);
}
```

- [ ] **Step 2: 类型检查 + 构建**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc && npm run build:weapp
```

预期：`tsc` exit 0；`Compiled successfully`。

- [ ] **Step 3: 人工验证（需后端可达 + 真机或开发者工具）**

1. 列表点任一行 → 详情面板打开，显示全部非空字段的键值表
2. 切「说明书」→ 出现加载中 → PDF 打开（或给出可读错误）
3. 切「权利要求」→ 结构化列表（第 N 项 · 独立/从属）；后端无结构化时应打开 PDF
4. 点左上 `‹` → 回到列表

- [ ] **Step 4: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add frontend/weapp/src/components/ResultDetail && \
  git commit -m "feat(weapp): 详情面板三 tab——详情零请求，说明书/权利要求走既有接口"
```

---

### Task 7: 历史会话恢复结果入口

**Files:**
- Modify: `frontend/weapp/src/pages/chat/index.tsx`

**Interfaces:**
- Consumes: `resultsStore.get` / `resultsStore.load`（Task 2）、`MsgView.resultSet`（Task 4）
- Produces: 无新导出

**背景**：`selectSession` 现在把 `detail.messages` 映射成只有 `{role, content}` 的 `MsgView`（`index.tsx:309-315`），`set_id` 被丢掉，结果入口不会出现。

- [ ] **Step 1: 恢复 `resultSet`**

把 `selectSession` 里的映射：

```tsx
      const history: MsgView[] = (detail.messages || [])
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .map((m: ChatMsg) => ({
          role: m.role as 'user' | 'assistant',
          content: m.content || '',
        }))
```

替换为：

```tsx
      const history: MsgView[] = (detail.messages || [])
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .map((m: ChatMsg) => {
          const view: MsgView = {
            role: m.role as 'user' | 'assistant',
            content: m.content || '',
          }
          // 消息里存了 set_id 且本机还有这份结果 → 复原入口。
          // 换设备/清过缓存时本地没有，就不显示入口（而不是给个点不开的按钮）。
          const setId = String((m as any).set_id || '')
          if (setId) {
            const found = resultsStore.get(setId) || resultsStore.load(setId)
            if (found) view.resultSet = { setId, rowCount: found.rows.length }
          }
          return view
        })
```

- [ ] **Step 2: 类型检查 + 构建**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc && npm run build:weapp
```

- [ ] **Step 3: 人工验证**

1. 发一个产生结果表的提问 → 关闭小程序 → **重开**
2. 抽屉里切回该会话 → 结果入口仍在，点开能看列表，**行数 ≤ 40**（持久化裁剪）
3. 清掉小程序缓存后重开会话 → **不显示入口**

- [ ] **Step 4: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add frontend/weapp/src/pages/chat/index.tsx && \
  git commit -m "feat(weapp): 历史会话复原结果入口——按 set_id，本机无缓存则不显示"
```

---

## 收尾

- [ ] **全量验证**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run test && npm run tsc && npm run build:weapp
```

预期：测试全绿；`tsc` exit 0；`Compiled successfully`。

- [ ] **真机走一遍验收标准**

spec §10 的 11 条。开发者工具与真机在文件系统、`openDocument` 上行为不同——本仓库所有前端结论历来只来自 `tsc` + build，真机验证不可省。
