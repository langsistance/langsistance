function isWorkflowParams(params) {
  if (!params) return false
  if (typeof params === 'object') return params.type === 'workflow'
  try {
    return JSON.parse(params)?.type === 'workflow'
  } catch {
    return false
  }
}

export function getKnowledgeTypeBadge(type, lang = 'en', params) {
  const isWorkflow = Number(type || 1) === 2 || isWorkflowParams(params)
  return {
    className: `knowledge-type-badge ${isWorkflow ? 'workflow' : 'normal'}`,
    label: isWorkflow
      ? (lang === 'en' ? 'Composed' : '组合知识')
      : (lang === 'en' ? 'Normal' : '普通知识'),
  }
}
