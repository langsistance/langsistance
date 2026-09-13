import { test } from 'node:test'
import assert from 'node:assert/strict'
import { visibleStatusSteps } from './chatSession.js'

// 2026-09-13 生产反馈：有时 status 轨上所有项都显示已完成，正在跑的那条
// **不出现**，用户看不到进度。根因是渲染层的去重无条件过滤 —— 当最新那条
// running 的文案恰好与某个 agent 步骤相同，它也被隐藏了。

const step = (thought, status = 'done') => ({ round: 1, thought, status })
const status = (message, state) => ({ id: message, message, state })

test('hides a done status that duplicates an agent step thought', () => {
  const out = visibleStatusSteps(
    [status('第 1 步 · 正在调用「patent_search_dual」', 'done'),
     status('正在分析您的问题...', 'done')],
    [step('第 1 步 · 正在调用「patent_search_dual」')],
  )
  assert.deepEqual(out.map((s) => s.message), ['正在分析您的问题...'])
})

test('never hides the running status, even when it duplicates a thought', () => {
  const out = visibleStatusSteps(
    [status('正在分析您的问题...', 'done'),
     status('第 2 步 · 正在调用「patent_search_dual」', 'running')],
    [step('第 2 步 · 正在调用「patent_search_dual」', 'running')],
  )
  assert.equal(out.length, 2)
  assert.equal(out.at(-1).state, 'running')
})

test('keeps every status when there are no agent steps', () => {
  const all = [status('a', 'done'), status('b', 'running')]
  assert.deepEqual(visibleStatusSteps(all, []), all)
  assert.deepEqual(visibleStatusSteps(all, undefined), all)
})

test('tolerates malformed entries', () => {
  const out = visibleStatusSteps(
    [null, undefined, status('x', 'running')],
    [step('x'), null],
  )
  assert.equal(out.length, 1)
})
