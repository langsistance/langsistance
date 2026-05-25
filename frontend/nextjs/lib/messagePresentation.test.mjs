import test from 'node:test'
import assert from 'node:assert/strict'

import { shouldShowAssistantWaiting } from './messagePresentation.js'

test('shows the assistant waiting indicator only while streaming before content arrives', () => {
  assert.equal(shouldShowAssistantWaiting('', true), true)
  assert.equal(shouldShowAssistantWaiting('   ', true), true)
  assert.equal(shouldShowAssistantWaiting('partial answer', true), false)
  assert.equal(shouldShowAssistantWaiting('', false), false)
})
