export function createChatId() {
  return Math.random().toString(36).slice(2)
}

export function createChatMessage(role, content = '') {
  return {
    id: createChatId(),
    role,
    content,
    artifacts: [],
  }
}

export function updateAssistantMessage(messages, messageId, contentDelta) {
  return messages.map((msg) =>
    msg.id === messageId
      ? { ...msg, content: msg.content + contentDelta }
      : msg
  )
}

export function replaceAssistantMessage(messages, messageId, newContent) {
  return messages.map((msg) =>
    msg.id === messageId
      ? { ...msg, content: newContent }
      : msg
  )
}

export function addAssistantArtifactStart(messages, messageId, event) {
  const artifactId = event.artifact_id || event.artifactId
  if (!artifactId) return messages

  const artifact = {
    artifactId,
    format: event.format,
    filename: event.filename,
    mimeType: event.mime_type || event.mimeType,
    rowCount: event.row_count || event.rowCount || 0,
    columnCount: event.column_count || event.columnCount || 0,
    chunks: [],
    complete: false,
  }

  return messages.map((msg) => {
    if (msg.id !== messageId) return msg
    const artifacts = Array.isArray(msg.artifacts) ? msg.artifacts : []
    return {
      ...msg,
      artifacts: [
        ...artifacts.filter((item) => item.artifactId !== artifactId),
        artifact,
      ],
    }
  })
}

export function addAssistantArtifactChunk(messages, messageId, artifactId, data) {
  if (!artifactId || !data) return messages

  return messages.map((msg) => {
    if (msg.id !== messageId) return msg
    const artifacts = Array.isArray(msg.artifacts) ? msg.artifacts : []
    return {
      ...msg,
      artifacts: artifacts.map((artifact) =>
        artifact.artifactId === artifactId
          ? { ...artifact, chunks: [...(artifact.chunks || []), data] }
          : artifact
      ),
    }
  })
}

export function addAssistantArtifactEnd(messages, messageId, artifactId) {
  if (!artifactId) return messages

  return messages.map((msg) => {
    if (msg.id !== messageId) return msg
    const artifacts = Array.isArray(msg.artifacts) ? msg.artifacts : []
    return {
      ...msg,
      artifacts: artifacts.map((artifact) =>
        artifact.artifactId === artifactId
          ? { ...artifact, complete: true }
          : artifact
      ),
    }
  })
}

/**
 * Attach hidden patent_ids to the assistant message so follow-up
 * conversation_refs queries include them in conversation_history.
 */
export function addAssistantPatentIds(messages, messageId, patentIds) {
  if (!patentIds || !Array.isArray(patentIds) || patentIds.length === 0) return messages
  return messages.map((msg) =>
    msg.id === messageId
      ? { ...msg, patent_ids: patentIds }
      : msg
  )
}

function base64ChunksToText(chunks) {
  // Decode each chunk SEPARATELY and merge bytes.  The backend slices at
  // 32768 bytes (32768 % 3 = 2), so every full chunk's base64 ends with
  // '=' padding — concatenating the base64 strings and decoding once
  // truncates at the first '='.  Per-chunk decoding mirrors the proven
  // base64ChunksToBlob download path.
  try {
    const byteArrays = chunks.map((chunk) => {
      if (typeof Buffer !== 'undefined') {
        return new Uint8Array(Buffer.from(chunk, 'base64'))
      }
      const binary = window.atob(chunk)
      const bytes = new Uint8Array(binary.length)
      for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
      return bytes
    })
    const total = byteArrays.reduce((sum, bytes) => sum + bytes.length, 0)
    const merged = new Uint8Array(total)
    let offset = 0
    for (const bytes of byteArrays) {
      merged.set(bytes, offset)
      offset += bytes.length
    }
    return new TextDecoder().decode(merged)
  } catch {
    return null
  }
}

/**
 * True when the message with *messageId* carries decoded results —
 * used to decide auto-navigation to the results page after streaming.
 */
export function hasResultsForMessage(messages, messageId) {
  const message = (Array.isArray(messages) ? messages : []).find((msg) => msg.id === messageId)
  return Boolean(message && message.results)
}

/**
 * Decode raw base64 artifact chunks into a results payload — the pure
 * decode step shared by decodeResultsArtifact (state path) and the chat
 * stream hook (synchronous navigation path).  Returns null on any failure.
 */
export function decodeArtifactChunksToResults(chunks, artifactId) {
  if (!Array.isArray(chunks) || chunks.length === 0) return null
  const text = base64ChunksToText(chunks)
  if (!text) return null
  try {
    const payload = JSON.parse(text)
    if (!payload || typeof payload !== 'object' || !Array.isArray(payload.rows)) {
      return null
    }
    return {
      setId: artifactId,
      source: payload.source || 'uspto',
      columns: Array.isArray(payload.columns) ? payload.columns : [],
      rows: payload.rows,
    }
  } catch {
    return null
  }
}

/**
 * Decode a complete format=json artifact into message.results.
 * Idempotent — returns messages unchanged when there is nothing to decode
 * (or the payload is malformed), so it is safe to call on every update.
 */
export function decodeResultsArtifact(messages, messageId) {
  return messages.map((msg) => {
    if (msg.id !== messageId || msg.results) return msg
    const artifacts = Array.isArray(msg.artifacts) ? msg.artifacts : []
    const jsonArtifact = artifacts.find(
      (artifact) => artifact.format === 'json' && artifact.complete,
    )
    if (!jsonArtifact) return msg
    const results = decodeArtifactChunksToResults(
      jsonArtifact.chunks || [],
      jsonArtifact.artifactId,
    )
    if (!results) return msg
    return { ...msg, results }
  })
}
