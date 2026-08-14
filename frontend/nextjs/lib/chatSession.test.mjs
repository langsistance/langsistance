import test from 'node:test'
import assert from 'node:assert/strict'

import {
  addAssistantArtifactChunk,
  addAssistantArtifactEnd,
  addAssistantArtifactStart,
  createChatId,
  createChatMessage,
  decodeArtifactChunksToResults,
  decodeResultsArtifact,
  hasResultsForMessage,
  shouldResetConversationOnNavigation,
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

test('decodes complete JSON artifact into message.results', () => {
  const assistant = createChatMessage('assistant', 'answer')
  const messages = [assistant]
  const payload = JSON.stringify({
    source: 'uspto',
    columns: [{ key: 'patentTitle', label: '标题', role: 'title' }],
    rows: [{ patentTitle: '一种图像处理方法' }],
  })
  const b64 = Buffer.from(payload, 'utf-8').toString('base64')

  let withArtifact = addAssistantArtifactStart(messages, assistant.id, {
    artifact_id: 'art-json',
    format: 'json',
    filename: 'r.json',
    mime_type: 'application/json',
    row_count: 1,
    column_count: 1,
  })
  withArtifact = addAssistantArtifactChunk(withArtifact, assistant.id, 'art-json', b64)
  withArtifact = addAssistantArtifactEnd(withArtifact, assistant.id, 'art-json')

  const decoded = decodeResultsArtifact(withArtifact, assistant.id)
  assert.ok(decoded[0].results)
  assert.equal(decoded[0].results.setId, 'art-json')
  assert.equal(decoded[0].results.source, 'uspto')
  assert.equal(decoded[0].results.rows[0].patentTitle, '一种图像处理方法')
})

test('decodeResultsArtifact leaves message untouched when no JSON artifact', () => {
  const assistant = createChatMessage('assistant', 'answer')
  const messages = [assistant]

  const withCsv = addAssistantArtifactStart(messages, assistant.id, {
    artifact_id: 'art-csv', format: 'csv', filename: 'r.csv',
  })
  const decoded = decodeResultsArtifact(withCsv, assistant.id)
  assert.equal(decoded[0].results, undefined)
})

test('decodeResultsArtifact survives malformed JSON', () => {
  const assistant = createChatMessage('assistant', 'answer')
  let messages = addAssistantArtifactStart([assistant], assistant.id, {
    artifact_id: 'art-bad', format: 'json', filename: 'r.json',
  })
  messages = addAssistantArtifactChunk(messages, assistant.id, 'art-bad', '%%%%')
  messages = addAssistantArtifactEnd(messages, assistant.id, 'art-bad')

  const decoded = decodeResultsArtifact(messages, assistant.id)
  assert.equal(decoded[0].results, undefined)
})

test('decodes multi-chunk artifacts chunked at 32768 bytes (padding trap regression)', () => {
  const assistant = createChatMessage('assistant', 'answer')
  const messages = [assistant]
  const payload = JSON.stringify({
    source: 'uspto',
    columns: [{ key: 'patentTitle', label: '标题', role: 'title' }],
    rows: Array.from({ length: 50 }, (_, i) => ({
      patentTitle: `Patent ${i} — `.repeat(200),
    })),
  })
  // Backend chunks at ARTIFACT_CHUNK_BYTES = 32768; 32768 % 3 = 2, so every
  // full chunk's base64 ends with '=' padding — concatenating padded
  // base64 and decoding once truncates at the first '='.
  const bytes = Buffer.from(payload, 'utf-8')
  const CHUNK = 32768
  const chunks = []
  for (let start = 0; start < bytes.length; start += CHUNK) {
    chunks.push(bytes.subarray(start, start + CHUNK).toString('base64'))
  }
  assert.ok(chunks.length > 1, 'payload must span multiple backend chunks')
  assert.ok(chunks[0].endsWith('='), 'first chunk must carry padding like production')

  let withArtifact = addAssistantArtifactStart(messages, assistant.id, {
    artifact_id: 'art-json-multi', format: 'json', filename: 'r.json',
    mime_type: 'application/json', row_count: 50, column_count: 1,
  })
  for (const chunk of chunks) {
    withArtifact = addAssistantArtifactChunk(withArtifact, assistant.id, 'art-json-multi', chunk)
  }
  withArtifact = addAssistantArtifactEnd(withArtifact, assistant.id, 'art-json-multi')

  const decoded = decodeResultsArtifact(withArtifact, assistant.id)
  assert.ok(decoded[0].results, 'multi-chunk JSON artifact must decode')
  assert.equal(decoded[0].results.rows.length, 50)
  assert.equal(decoded[0].results.rows[49].patentTitle.startsWith('Patent 49'), true)
})

test('hasResultsForMessage detects attached results by message id', () => {
  const assistant = createChatMessage('assistant', 'answer')
  let messages = addAssistantArtifactStart([assistant], assistant.id, {
    artifact_id: 'art-json-x', format: 'json', filename: 'r.json',
  })
  messages = addAssistantArtifactChunk(
    messages, assistant.id, 'art-json-x',
    Buffer.from(JSON.stringify({ source: 'uspto', columns: [], rows: [{}] }), 'utf-8').toString('base64'),
  )
  messages = addAssistantArtifactEnd(messages, assistant.id, 'art-json-x')
  messages = decodeResultsArtifact(messages, assistant.id)

  assert.equal(hasResultsForMessage(messages, assistant.id), true)
  assert.equal(hasResultsForMessage(messages, 'other-id'), false)
})

test('hasResultsForMessage returns false without results', () => {
  const assistant = createChatMessage('assistant', 'plain answer')
  assert.equal(hasResultsForMessage([assistant], assistant.id), false)
})

test('decodeArtifactChunksToResults decodes chunks into a results payload', () => {
  const payload = { source: 'uspto', columns: [], rows: [{ patentTitle: 'T' }] }
  const b64 = Buffer.from(JSON.stringify(payload), 'utf-8').toString('base64')
  const results = decodeArtifactChunksToResults([b64], 'art-1')
  assert.equal(results.setId, 'art-1')
  assert.equal(results.source, 'uspto')
  assert.equal(results.rows.length, 1)
})

test('decodeArtifactChunksToResults returns null for malformed data', () => {
  assert.equal(decodeArtifactChunksToResults(['%%%%'], 'art-1'), null)
  assert.equal(decodeArtifactChunksToResults([], 'art-1'), null)
  assert.equal(decodeArtifactChunksToResults(['e30='], 'art-1'), null) // {} — no rows
})

test('shouldResetConversationOnNavigation only resets on the chat route', () => {
  assert.equal(shouldResetConversationOnNavigation('/app/chat'), true)
  // trailingSlash: true — usePathname() returns the slash variant
  assert.equal(shouldResetConversationOnNavigation('/app/chat/'), true)
  // The results page auto-opens after a search with no session_id — the
  // transition must not wipe the shared conversation (regression: the chat
  // page's session-load effect re-renders with the new URL before unmount).
  assert.equal(shouldResetConversationOnNavigation('/app/results'), false)
  assert.equal(shouldResetConversationOnNavigation('/app/results/'), false)
  assert.equal(shouldResetConversationOnNavigation('/app/knowledge'), false)
  assert.equal(shouldResetConversationOnNavigation('/app'), false)
})
