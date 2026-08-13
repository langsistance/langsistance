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
