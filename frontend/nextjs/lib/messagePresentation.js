export function shouldShowAssistantWaiting(content, streaming) {
  return Boolean(streaming && !String(content || '').trim())
}

export function shouldShowAssistantTransientStatus(status, streaming) {
  return Boolean(streaming && String(status || '').trim())
}
