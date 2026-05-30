import test from 'node:test'
import assert from 'node:assert/strict'

import { getKnowledgeTypeBadge } from './knowledgeTypeBadge.js'

test('knowledge type badge labels normal knowledge', () => {
  assert.deepEqual(getKnowledgeTypeBadge(1, 'en'), {
    className: 'knowledge-type-badge normal',
    label: 'Normal',
  })
  assert.deepEqual(getKnowledgeTypeBadge(undefined, 'zh'), {
    className: 'knowledge-type-badge normal',
    label: '普通知识',
  })
})

test('knowledge type badge labels composed knowledge', () => {
  assert.deepEqual(getKnowledgeTypeBadge(2, 'en'), {
    className: 'knowledge-type-badge workflow',
    label: 'Composed',
  })
  assert.deepEqual(getKnowledgeTypeBadge('2', 'zh'), {
    className: 'knowledge-type-badge workflow',
    label: '组合知识',
  })
})

test('knowledge type badge infers composed knowledge from workflow params', () => {
  assert.deepEqual(getKnowledgeTypeBadge(undefined, 'en', '{"type":"workflow","steps":[]}'), {
    className: 'knowledge-type-badge workflow',
    label: 'Composed',
  })
  assert.deepEqual(getKnowledgeTypeBadge(1, 'zh', { type: 'workflow', steps: [] }), {
    className: 'knowledge-type-badge workflow',
    label: '组合知识',
  })
})
