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
  try {
    if (typeof Buffer !== 'undefined') {
      return Buffer.from(chunks.join(''), 'base64').toString('utf-8')
    }
    const binary = window.atob(chunks.join(''))
    const bytes = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
    return new TextDecoder().decode(bytes)
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
    const text = base64ChunksToText(jsonArtifact.chunks || [])
    if (!text) return msg
    try {
      const payload = JSON.parse(text)
      if (!payload || typeof payload !== 'object' || !Array.isArray(payload.rows)) {
        return msg
      }
      return {
        ...msg,
        results: {
          setId: jsonArtifact.artifactId,
          source: payload.source || 'uspto',
          columns: Array.isArray(payload.columns) ? payload.columns : [],
          rows: payload.rows,
        },
      }
    } catch {
      return msg
    }
  })
}
