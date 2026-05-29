import test from 'node:test'
import assert from 'node:assert/strict'

import {
  shouldShowAssistantTransientStatus,
  shouldShowAssistantWaiting,
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
