import test from 'node:test'
import assert from 'node:assert/strict'
// 必须排在最前：补上 Taro 构建期才会 inlined 的裸标识符常量，
// 否则下一行 import resultsStore → @tarojs/taro → @tarojs/runtime 会直接 ReferenceError。
import './taroRuntimeStubs.js'
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

/**
 * 上面那条用例的插入序与 savedAt 序**一致**，所以「按 savedAt 淘汰」与
 * 「按对象键序淘汰」在它眼里没有区别——把实现换成键序照样绿。
 * 这条把插入序与 savedAt 序**反过来**，两种实现才会分道扬镳：
 * 键序实现在这里会丢掉最后插入的（savedAt 最新的）那条。
 */
test('淘汰按 savedAt：插入序与 savedAt 序相反时，丢的仍是最旧的 savedAt', () => {
  const storage = fakeStorage()
  const store = createResultsStore(storage)
  const total = MAX_RESULT_SETS + 2
  // 先插最新、再插最旧 —— 对象键序与 savedAt 序相反
  for (let i = total - 1; i >= 0; i--) {
    store.persist(payload(`s-${i}`), { sessionId: 's', queryText: 'q', savedAt: 1000 + i })
  }
  const ids = Object.keys(storage.getSync(STORAGE_KEY).sets)
  assert.equal(ids.length, MAX_RESULT_SETS)
  assert.ok(!ids.includes('s-0'), 'savedAt 最旧的 s-0 没被丢——说明淘汰看的是键序')
  assert.ok(!ids.includes('s-1'))
  assert.ok(ids.includes('s-2'), '边界丢多了')
  assert.ok(ids.includes(`s-${total - 1}`), 'savedAt 最新的那条被丢了')
})

test('写盘失败静默放弃，不抛——配额问题不能打断对话', () => {
  const store = createResultsStore(fakeStorage({ failOnWrite: true }))
  assert.doesNotThrow(() => store.persist(payload('a'), { sessionId: 's', queryText: 'q' }))
  // 内存仍然可用
  assert.equal(store.get('a').rows.length, 3)
})
