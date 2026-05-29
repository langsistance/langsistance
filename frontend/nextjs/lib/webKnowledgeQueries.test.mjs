import test from 'node:test'
import assert from 'node:assert/strict'

import {
  WEB_KNOWLEDGE_PUSH_FILTER,
  withWebKnowledgePushFilter,
} from './webKnowledgeQueries.js'

test('web knowledge queries always request push=2 knowledge', () => {
  assert.equal(WEB_KNOWLEDGE_PUSH_FILTER, 2)
  assert.deepEqual(
    withWebKnowledgePushFilter({ query: 'demo', offset: 0, limit: 12 }),
    { query: 'demo', offset: 0, limit: 12, push_filter: 2 }
  )
})

test('web knowledge push filter cannot be overridden by callers', () => {
  assert.deepEqual(
    withWebKnowledgePushFilter({ push_filter: 1, limit: 10 }),
    { push_filter: 2, limit: 10 }
  )
})
