import test from 'node:test'
import assert from 'node:assert/strict'

import {
  failureHint,
  sanitizeLegacyMarkers,
  shouldShowAssistantTransientStatus,
  shouldShowAssistantWaiting,
  shouldShowStatusSteps,
} from './messagePresentation.js'

test('shows the assistant waiting indicator only while streaming before content arrives', () => {
  assert.equal(shouldShowAssistantWaiting('', true), true)
  assert.equal(shouldShowAssistantWaiting('   ', true), true)
  assert.equal(shouldShowAssistantWaiting('partial answer', true), false)
  assert.equal(shouldShowAssistantWaiting('', false), false)
})

test('shows transient assistant status while streaming even after content arrives', () => {
  assert.equal(shouldShowAssistantTransientStatus('Filtering results 1-5 of 12...', true), true)
  assert.equal(shouldShowAssistantTransientStatus('Filtering results 1-5 of 12...', false), false)
  assert.equal(shouldShowAssistantTransientStatus('', true), false)
  assert.equal(shouldShowAssistantTransientStatus('   ', true), false)
})

test('shows status steps only on the streaming message, never on historical cards', () => {
  const steps = [
    { id: 1, message: 'Filtering results 1-5 of 12...', state: 'running' },
    { id: 2, message: 'Retrieving 6 more', state: 'done' },
  ]
  // Current streaming message: steps visible
  assert.equal(shouldShowStatusSteps(steps, true), true)
  // Historical assistant card (streaming=false): steps must NOT leak onto it
  assert.equal(shouldShowStatusSteps(steps, false), false)
  // No steps / empty list: nothing to show either way
  assert.equal(shouldShowStatusSteps([], true), false)
  assert.equal(shouldShowStatusSteps(undefined, true), false)
})

// ── 需求 4: 存量内部标记可读化 + 失败分类建议 ──

test('sanitizeLegacyMarkers replaces the internal marker with a user-facing hint', () => {
  const raw = '结果如下。\n<Knowledge tool not logged in>'
  const out = sanitizeLegacyMarkers(raw)
  assert.ok(!out.includes('<Knowledge tool not logged in>'))
  assert.ok(out.includes('该工具需要登录后才能使用'))
})

test('sanitizeLegacyMarkers leaves normal content untouched', () => {
  const raw = '正常回答内容'
  assert.equal(sanitizeLegacyMarkers(raw), raw)
  assert.equal(sanitizeLegacyMarkers(''), '')
  assert.equal(sanitizeLegacyMarkers(null), null)
})

test('failureHint maps no_patents_found to actionable alternatives', () => {
  const hint = failureHint('未找到匹配的专利')
  assert.ok(hint.includes('更换关键词'))
  assert.ok(hint.includes('专利号'))
})

test('failureHint maps 403 to authorization guidance', () => {
  const hint = failureHint('USPTO HTTP 403')
  assert.ok(hint.includes('授权'))
})

test('failureHint returns null for unknown errors', () => {
  assert.equal(failureHint('random error text'), null)
  assert.equal(failureHint(''), null)
})
