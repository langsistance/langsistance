export function shouldShowAssistantWaiting(content, streaming) {
  return Boolean(streaming && !String(content || '').trim())
}

export function shouldShowAssistantTransientStatus(status, streaming) {
  return Boolean(streaming && String(status || '').trim())
}

export function shouldShowStatusSteps(statusSteps, streaming) {
  return Boolean(streaming && Array.isArray(statusSteps) && statusSteps.length > 0)
}

/**
 * 需求 4: 存量消息中的内部标记可读化。
 *
 * 旧版本 system prompt 曾指示 LLM 在工具未认证时输出
 * `<Knowledge tool not logged in>` 标记; 新 prompt 已禁止, 但历史消息
 * 里仍可能出现。渲染前替换为面向用户的可操作提示。
 */
export const LEGACY_NOT_LOGGED_IN_MARKER = '<Knowledge tool not logged in>'
export const NOT_LOGGED_IN_HINT =
  '> ⚠️ 该工具需要登录后才能使用，请先完成登录再重试。'

export function sanitizeLegacyMarkers(text) {
  if (!text || !text.includes(LEGACY_NOT_LOGGED_IN_MARKER)) return text
  return text.split(LEGACY_NOT_LOGGED_IN_MARKER).join(NOT_LOGGED_IN_HINT)
}

/**
 * 需求 4: 失败消息按错误类型给出可操作的替代路径建议。
 */
export function failureHint(errorMessage) {
  if (!errorMessage) return null
  const text = String(errorMessage).toLowerCase()
  if (text.includes('no_patents_found') || text.includes('未找到匹配') || text.includes('未识别到专利号')) {
    return '可尝试：更换关键词 / 提供专利号 / 上传专利文件。'
  }
  if (text.includes('403') || text.includes('未授权') || text.includes('not logged in') || text.includes('登录')) {
    return '数据源需要授权，请确认登录与工具凭据后重试。'
  }
  return null
}
