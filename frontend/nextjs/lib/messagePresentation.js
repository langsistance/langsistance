export function shouldShowAssistantWaiting(content, streaming) {
  return Boolean(streaming && !String(content || '').trim())
}
