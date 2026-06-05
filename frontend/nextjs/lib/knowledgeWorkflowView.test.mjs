import test from 'node:test'
import assert from 'node:assert/strict'

import {
  getKnowledgeNamesByIdFromExtraInfo,
  getWorkflowInstructionsForReadOnly,
  getWorkflowStepLabels,
  isWorkflowKnowledgeItem,
  parseWorkflowSteps,
} from './knowledgeWorkflowView.js'

test('recognizes composed knowledge from type or workflow params', () => {
  assert.equal(isWorkflowKnowledgeItem({ type: 2 }), true)
  assert.equal(isWorkflowKnowledgeItem({ type: 1, params: '{"type":"workflow","steps":[]}' }), true)
  assert.equal(isWorkflowKnowledgeItem({ type: 1, params: '{}' }), false)
})

test('parses workflow steps from JSON params', () => {
  assert.deepEqual(
    parseWorkflowSteps('{"type":"workflow","steps":[{"knowledge_id":101},{"knowledge_id":"102"}]}'),
    [
      { id: 'step_1', knowledgeId: 101 },
      { id: 'step_2', knowledgeId: 102 },
    ]
  )
})

test('returns no steps for normal or invalid workflow params', () => {
  assert.deepEqual(parseWorkflowSteps('{"type":"normal","steps":[{"knowledge_id":101}]}'), [])
  assert.deepEqual(parseWorkflowSteps('not json'), [])
})

test('formats workflow step labels with fallback IDs', () => {
  assert.deepEqual(
    getWorkflowStepLabels(
      '{"type":"workflow","steps":[{"knowledge_id":101},{"knowledge_id":102}]}',
      { 101: 'Find patents' },
      'en'
    ),
    ['Find patents', 'Knowledge #102']
  )
})

test('maps workflow dependency metadata to knowledge names by id', () => {
  assert.deepEqual(
    getKnowledgeNamesByIdFromExtraInfo({
      workflow_dependencies: [
        { knowledge_id: 101, question: 'Find patents by publication ID' },
        { knowledge_id: '102', question: 'Download patent PDF' },
        { knowledge_id: 103, question: '' },
        { knowledge_id: 'bad', question: 'Ignored' },
      ],
    }),
    {
      101: 'Find patents by publication ID',
      102: 'Download patent PDF',
    }
  )
})

test('hides generated workflow instructions in read-only detail view', () => {
  assert.equal(
    getWorkflowInstructionsForReadOnly('Composed knowledge: execute 2 knowledge steps in order.', 2),
    ''
  )
})

test('keeps custom workflow instructions in read-only detail view', () => {
  assert.equal(
    getWorkflowInstructionsForReadOnly('Use the first step result to query the next step.', 2),
    'Use the first step result to query the next step.'
  )
})
