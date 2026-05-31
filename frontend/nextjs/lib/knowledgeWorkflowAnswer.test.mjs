import test from 'node:test'
import assert from 'node:assert/strict'

import {
  getKnowledgeWorkflowAnswer,
  getWorkflowInstructionsEditorValue,
  getWorkflowInstructionsForSave,
} from './knowledgeWorkflowAnswer.js'

test('formats composed knowledge answer in English', () => {
  assert.equal(
    getKnowledgeWorkflowAnswer(3, 'en'),
    'Composed knowledge: execute 3 knowledge steps in order.'
  )
})

test('formats composed knowledge answer in Chinese', () => {
  assert.equal(
    getKnowledgeWorkflowAnswer(2, 'zh'),
    '组合知识：按顺序执行 2 个知识步骤。'
  )
})

test('falls back to English for unsupported languages', () => {
  assert.equal(
    getKnowledgeWorkflowAnswer(4, 'fr'),
    'Composed knowledge: execute 4 knowledge steps in order.'
  )
})

test('uses custom workflow instructions when provided', () => {
  assert.equal(
    getWorkflowInstructionsForSave('  Use the user request to drive each step.  ', 2, 'en'),
    'Use the user request to drive each step.'
  )
})

test('falls back to default workflow answer when instructions are blank', () => {
  assert.equal(
    getWorkflowInstructionsForSave('   ', 2, 'en'),
    'Composed knowledge: execute 2 knowledge steps in order.'
  )
})

test('refreshes a generated default workflow answer when step count changes', () => {
  assert.equal(
    getWorkflowInstructionsForSave('Composed knowledge: execute 2 knowledge steps in order.', 3, 'en'),
    'Composed knowledge: execute 3 knowledge steps in order.'
  )
})

test('hides generated default workflow answer in the editor', () => {
  assert.equal(
    getWorkflowInstructionsEditorValue('Composed knowledge: execute 2 knowledge steps in order.', 2),
    ''
  )
})

test('hides generated Chinese default workflow answer in the editor', () => {
  assert.equal(
    getWorkflowInstructionsEditorValue('组合知识：按顺序执行 2 个知识步骤。', 3),
    ''
  )
})

test('preserves custom workflow instructions in the editor', () => {
  assert.equal(
    getWorkflowInstructionsEditorValue('Use the first step result to query the second step.', 2),
    'Use the first step result to query the second step.'
  )
})
