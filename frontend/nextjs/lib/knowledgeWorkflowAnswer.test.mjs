import test from 'node:test'
import assert from 'node:assert/strict'

import { getKnowledgeWorkflowAnswer } from './knowledgeWorkflowAnswer.js'

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
