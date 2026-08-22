export function shouldShowAssistantWaiting(content, streaming) {
  return Boolean(streaming && !String(content || '').trim())
}

export function shouldShowAssistantTransientStatus(status, streaming) {
  return Boolean(streaming && String(status || '').trim())
}

export function shouldShowStatusSteps(statusSteps, streaming) {
  return Boolean(streaming && Array.isArray(statusSteps) && statusSteps.length > 0)
}
