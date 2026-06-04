import test from 'node:test'
import assert from 'node:assert/strict'

import {
  addAssistantArtifactChunk,
  addAssistantArtifactEnd,
  addAssistantArtifactStart,
  createChatId,
  createChatMessage,
  updateAssistantMessage,
} from './chatSession.js'

test('chat session creates non-empty ids for requests and messages', () => {
  const id = createChatId()

  assert.equal(typeof id, 'string')
  assert.ok(id.length > 0)
})

test('chat session creates stable message records for layout-level state', () => {
  const message = createChatMessage('user', 'hello')

  assert.equal(message.role, 'user')
  assert.equal(message.content, 'hello')
  assert.equal(typeof message.id, 'string')
  assert.ok(message.id.length > 0)
})

test('chat session appends streamed assistant content by id', () => {
  const assistant = createChatMessage('assistant', 'he')
  const messages = [
    createChatMessage('user', 'question'),
    assistant,
  ]

  const updated = updateAssistantMessage(messages, assistant.id, 'llo')

  assert.equal(updated[1].content, 'hello')
  assert.notEqual(updated, messages)
  assert.equal(messages[1].content, 'he')
})

test('chat session stores streamed assistant artifacts by id', () => {
  const assistant = createChatMessage('assistant', 'answer')
  const messages = [assistant]

  const withArtifact = addAssistantArtifactStart(messages, assistant.id, {
    artifact_id: 'artifact-1',
    format: 'csv',
    filename: 'results.csv',
    mime_type: 'text/csv;charset=utf-8',
    row_count: 6,
    column_count: 3,
  })
  const withChunk = addAssistantArtifactChunk(
    withArtifact,
    assistant.id,
    'artifact-1',
    'YmFzZTY0'
  )
  const complete = addAssistantArtifactEnd(withChunk, assistant.id, 'artifact-1')

  assert.equal(complete[0].artifacts.length, 1)
  assert.equal(complete[0].artifacts[0].format, 'csv')
  assert.deepEqual(complete[0].artifacts[0].chunks, ['YmFzZTY0'])
  assert.equal(complete[0].artifacts[0].complete, true)
})
