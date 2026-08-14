import test from 'node:test'
import assert from 'node:assert/strict'

import { resultsPath, chatPath } from './appRoutes.js'

test('resultsPath always carries the trailing slash', () => {
  // Regression: bare /app/results?set=… made the static-export client
  // fetch /app/results.txt (404) and fall back to a full-page navigation,
  // wiping the in-memory conversation.
  assert.equal(resultsPath('s1', null), '/app/results/?set=s1')
  assert.equal(resultsPath('s1', 'sid-1'), '/app/results/?set=s1&session_id=sid-1')
  assert.ok(resultsPath('s1', null).startsWith('/app/results/'))
})

test('chatPath always carries the trailing slash', () => {
  assert.equal(chatPath(), '/app/chat/')
  assert.equal(chatPath('session_id=s1'), '/app/chat/?session_id=s1')
  assert.equal(chatPath('session_id=s1&pending_query=x%20y'), '/app/chat/?session_id=s1&pending_query=x%20y')
  assert.ok(chatPath('session_id=s1').startsWith('/app/chat/'))
})
