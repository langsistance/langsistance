import test from 'node:test'
import assert from 'node:assert/strict'

import { filterKnowledgeBaseTools } from './toolFilters.js'

test('knowledge base tool list only includes Push=2 tools', () => {
  const tools = [
    { id: 1, title: 'push 1', push: 1 },
    { id: 2, title: 'push 2', push: 2 },
    { id: 3, title: 'push 3', push: 3 },
  ]

  assert.deepEqual(filterKnowledgeBaseTools(tools), [
    { id: 2, title: 'push 2', push: 2 },
  ])
})
