import test from 'node:test'
import assert from 'node:assert/strict'

import { KNOWLEDGE_LIST_PAGE_SIZE } from './appUiConfig.js'

test('knowledge and community lists request a complete two-row page', () => {
  assert.equal(KNOWLEDGE_LIST_PAGE_SIZE, 12)
})
