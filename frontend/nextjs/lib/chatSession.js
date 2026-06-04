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
