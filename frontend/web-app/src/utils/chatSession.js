export function createChatId() {
  return Math.random().toString(36).slice(2)
}

export function createChatMessage(role, content = '') {
  return {
    id: createChatId(),
    role,
    content,
  }
}

export function updateAssistantMessage(messages, messageId, contentDelta) {
  return messages.map((msg) =>
    msg.id === messageId
      ? { ...msg, content: msg.content + contentDelta }
      : msg
  )
}
