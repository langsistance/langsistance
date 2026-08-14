import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  applyAgentStep,
  applyAgentObservation,
  applyAgentElapsed,
} from './chatSession.js'
import { pruneMessagesForPersistence } from './chatStore.js'

const msg = { id: 'm1', role: 'assistant', content: '' }

test('applyAgentStep appends a running step', () => {
  const out = applyAgentStep([msg], 'm1', {
    round: 1, thought: '第 1 步', action: 'a',
    params_brief: '{}', reasoning_text: 'r',
  })
  assert.equal(out[0].agentSteps.length, 1)
  assert.equal(out[0].agentSteps[0].status, 'running')
  assert.equal(out[0].agentSteps[0].reasoningText, 'r')
  assert.equal(out[0].agentSteps[0].paramsBrief, '{}')
})

test('applyAgentStep merges into an existing round', () => {
  const once = applyAgentStep([msg], 'm1', { round: 1, thought: 'a', action: 'x' })
  const twice = applyAgentStep(once, 'm1', { round: 1, thought: 'b', action: 'y' })
  assert.equal(twice[0].agentSteps.length, 1)
  assert.equal(twice[0].agentSteps[0].thought, 'b')
  assert.equal(twice[0].agentSteps[0].action, 'y')
})

test('applyAgentObservation marks the step done', () => {
  const once = applyAgentStep([msg], 'm1', { round: 1, action: 'a' })
  const out = applyAgentObservation(once, 'm1', { round: 1, result_brief: 'ok' })
  assert.equal(out[0].agentSteps[0].status, 'done')
  assert.equal(out[0].agentSteps[0].observationBrief, 'ok')
})

test('applyAgentElapsed sets elapsedSeconds and closes running steps', () => {
  const once = applyAgentStep([msg], 'm1', { round: 1, action: 'a' })
  const out = applyAgentElapsed(once, 'm1', { elapsed_seconds: 3.2, steps: 1 })
  assert.equal(out[0].elapsedSeconds, 3.2)
  assert.equal(out[0].agentSteps[0].status, 'done')
})

test('events target only the matching message', () => {
  const other = { id: 'm2', role: 'assistant', content: '' }
  const out = applyAgentStep([msg, other], 'm9', { round: 1, action: 'a' })
  assert.equal(out[0].agentSteps, undefined)
  assert.equal(out[1].agentSteps, undefined)
})

test('persistence keeps agentSteps and elapsedSeconds', () => {
  const out = pruneMessagesForPersistence([{
    id: 'm1', role: 'assistant', content: 'x',
    agentSteps: [{ round: 1, action: 'a', status: 'done' }],
    elapsedSeconds: 2,
  }])
  assert.equal(out[0].agentSteps.length, 1)
  assert.equal(out[0].elapsedSeconds, 2)
})
