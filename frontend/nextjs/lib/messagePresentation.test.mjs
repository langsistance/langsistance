import test from 'node:test'
import assert from 'node:assert/strict'

import {
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
