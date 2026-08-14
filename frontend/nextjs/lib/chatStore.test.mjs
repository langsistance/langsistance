import test from 'node:test'
import assert from 'node:assert/strict'

import {
  CHAT_STORE_KEY,
  MAX_PERSIST_SUMMARY_CHARS,
  loadChatStore,
  persistChatToStorage,
  pruneMessagesForPersistence,
  saveChatStore,
} from './chatStore.js'
import {
  loadResultsStore,
  persistResultsSetToStorage,
  restoreResultsInMessages,
} from './resultsStore.js'

function memoryStorage() {
  const map = new Map()
  return {
    getItem: (key) => (map.has(key) ? map.get(key) : null),
    setItem: (key, value) => { map.set(key, value) },
  }
}

const RESULTS = {
  setId: 'set-1',
  source: 'uspto',
  columns: [
    { key: 'patentTitle', label: '标题', role: 'title' },
    { key: 'abstractText', label: '摘要', role: 'abstract' },
    { key: 'customThing', label: '自定义', role: 'text' },
  ],
  rows: [{ patentTitle: '一种图像处理方法', abstractText: 'x'.repeat(700), customThing: 'x' }],
}

// The in-memory conversation as the landing page's ChatProvider holds it
// right after a search stream completes (artifact chunks still attached).
function landingConversation() {
  return [
    { id: 'u1', role: 'user', content: '搜索图像处理专利' },
    {
      id: 'a1',
      role: 'assistant',
      content: '为你找到以下专利。',
      artifacts: [
        {
          artifactId: 'set-1',
          format: 'json',
          filename: 'results.json',
          mimeType: 'application/json',
          rowCount: 1,
          columnCount: 3,
          chunks: ['<large base64 json chunk>'],
          complete: true,
        },
        {
          artifactId: 'dl-1',
          format: 'xlsx',
          filename: 'results.xlsx',
          mimeType: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
          rowCount: 1,
          columnCount: 3,
          chunks: ['<large base64 xlsx chunk>'],
          complete: true,
        },
      ],
      results: RESULTS,
      patent_ids: ['US12000123B2'],
      taskId: 'task-1',
      resultSummary: '# 分析报告\n\n生成完毕。',
    },
  ]
}

test('pruneMessagesForPersistence keeps artifact chunks and prunes results', () => {
  const pruned = pruneMessagesForPersistence(landingConversation())
  assert.equal(pruned.length, 2)
  assert.equal(pruned[0].id, 'u1')
  assert.equal(pruned[0].role, 'user')
  assert.equal(pruned[0].content, '搜索图像处理专利')
  const assistant = pruned[1]
  assert.equal(assistant.content, '为你找到以下专利。')
  // Download buttons depend on the csv/xlsx chunks surviving the remount;
  // the json artifact's chunks are dropped (its data lives in msg.results).
  assert.equal(assistant.artifacts.length, 2)
  assert.equal(assistant.artifacts[0].format, 'json')
  assert.deepEqual(assistant.artifacts[0].chunks, [])
  assert.equal(assistant.artifacts[1].format, 'xlsx')
  assert.equal(assistant.artifacts[1].chunks[0], '<large base64 xlsx chunk>')
  assert.deepEqual(assistant.patent_ids, ['US12000123B2'])
  assert.equal(assistant.taskId, 'task-1')
  assert.equal(assistant.resultSummary, '# 分析报告\n\n生成完毕。')
  assert.ok(assistant.results)
  assert.equal(assistant.results.rows[0].abstractText.length, 500) // abstract capped
  assert.equal(assistant.results.rows[0].customThing, undefined) // text role dropped
})

test('pruneMessagesForPersistence caps oversized result summaries', () => {
  const [pruned] = pruneMessagesForPersistence([
    { id: 'a1', role: 'assistant', content: 'x', resultSummary: 'y'.repeat(MAX_PERSIST_SUMMARY_CHARS + 100) },
  ])
  assert.equal(pruned.resultSummary.length, MAX_PERSIST_SUMMARY_CHARS)
})

test('pruneMessagesForPersistence drops malformed entries', () => {
  assert.deepEqual(pruneMessagesForPersistence(null), [])
  assert.deepEqual(pruneMessagesForPersistence([null, { id: 'x', role: 'user' }]), [])
})

test('loadChatStore returns [] when storage is missing, empty, or corrupt', () => {
  assert.deepEqual(loadChatStore(null), [])
  assert.deepEqual(loadChatStore(memoryStorage()), [])
  const corrupt = memoryStorage()
  corrupt.setItem(CHAT_STORE_KEY, '{not json')
  assert.deepEqual(loadChatStore(corrupt), [])
  const wrongShape = memoryStorage()
  wrongShape.setItem(CHAT_STORE_KEY, JSON.stringify({ not: 'an array' }))
  assert.deepEqual(loadChatStore(wrongShape), [])
})

test('persistChatToStorage round-trips a pruned conversation', () => {
  const storage = memoryStorage()
  persistChatToStorage(storage, landingConversation())
  const restored = loadChatStore(storage)
  assert.equal(restored.length, 2)
  assert.equal(restored[0].content, '搜索图像处理专利')
  assert.equal(restored[1].content, '为你找到以下专利。')
  assert.equal(restored[1].artifacts.length, 2)
  assert.deepEqual(restored[1].artifacts[0].chunks, []) // json chunks dropped
  assert.equal(restored[1].artifacts[1].chunks[0], '<large base64 xlsx chunk>')
  assert.ok(restored[1].results)
})

test('saveChatStore strips artifact chunks when quota is exceeded', () => {
  const stored = []
  const quotaStorage = {
    getItem: () => null,
    setItem: (key, value) => {
      if (JSON.parse(value)[1].artifacts.length > 0) {
        const error = new Error('quota')
        error.name = 'QuotaExceededError'
        throw error
      }
      stored.push(JSON.parse(value))
    },
  }
  persistChatToStorage(quotaStorage, landingConversation())
  assert.equal(stored.length, 1)
  assert.equal(stored[0][1].content, '为你找到以下专利。') // conversation survives
  assert.deepEqual(stored[0][1].artifacts, []) // only chunks degrade
})

test('saveChatStore degrades silently when storage throws', () => {
  const throwing = {
    getItem: () => { throw new Error('blocked') },
    setItem: () => { throw new Error('blocked') },
  }
  assert.doesNotThrow(() => saveChatStore(throwing, landingConversation()))
  assert.deepEqual(loadChatStore(throwing), [])
})

// Regression: the landing page (/) and /app/* routes each mount their own
// ChatProvider.  A client-side navigation to the results page unmounted the
// landing provider and its in-memory messages, so the question and answer
// disappeared.  The persisted copy must let the fresh provider rebuild the
// conversation exactly the way the chat page restores sessions.
test('conversation asked on the landing page survives a provider remount', () => {
  const sessionStorage = memoryStorage()
  const localStorage = memoryStorage()
  persistResultsSetToStorage(localStorage, RESULTS, {
    sessionId: null,
    queryText: '搜索图像处理专利',
    savedAt: 0,
  })
  persistChatToStorage(sessionStorage, landingConversation())

  // Fresh provider mount: hydrate the way ChatProvider/chat page do.
  const hydrated = restoreResultsInMessages(
    loadChatStore(sessionStorage),
    loadResultsStore(localStorage),
  )

  assert.equal(hydrated.length, 2)
  assert.equal(hydrated[0].content, '搜索图像处理专利') // the question survives
  assert.equal(hydrated[1].content, '为你找到以下专利。') // the answer survives
  assert.ok(hydrated[1].results)
  assert.equal(hydrated[1].results.setId, 'set-1')
})

test('hydration re-attaches results from the results store when the message copy has none', () => {
  const sessionStorage = memoryStorage()
  const localStorage = memoryStorage()
  persistResultsSetToStorage(localStorage, RESULTS, {
    sessionId: null,
    queryText: '搜索图像处理专利',
    savedAt: 0,
  })
  const withoutResults = landingConversation().map((m) => ({ ...m, results: undefined }))
  persistChatToStorage(sessionStorage, withoutResults)

  const hydrated = restoreResultsInMessages(
    loadChatStore(sessionStorage),
    loadResultsStore(localStorage),
  )

  assert.ok(hydrated[1].results)
  assert.equal(hydrated[1].results.setId, 'set-1')
})
