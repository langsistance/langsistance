import { getWorkflowInstructionsEditorValue } from './knowledgeWorkflowAnswer.js'

export function parseWorkflowParams(params) {
  if (!params) return null
  if (typeof params === 'object') return params
  try {
    return JSON.parse(params)
  } catch {
    return null
  }
}

export function isWorkflowKnowledgeItem(item) {
  if (!item) return false
  const params = parseWorkflowParams(item.params)
  return Number(item.type || 1) === 2 || params?.type === 'workflow'
}

export function parseWorkflowSteps(params) {
  const workflow = parseWorkflowParams(params)
  if (workflow?.type !== 'workflow' || !Array.isArray(workflow.steps)) {
    return []
  }

  return workflow.steps
    .map((step, index) => {
      const knowledgeId = Number(step?.knowledge_id)
      if (!Number.isFinite(knowledgeId) || knowledgeId <= 0) {
        return null
      }
      return {
        id: String(step?.id || `step_${index + 1}`),
        knowledgeId,
      }
    })
    .filter(Boolean)
}

export function getWorkflowStepLabels(params, knowledgeNamesById = {}, lang = 'en') {
  return parseWorkflowSteps(params).map((step) => (
    knowledgeNamesById[step.knowledgeId] ||
    (lang === 'en' ? `Knowledge #${step.knowledgeId}` : `知识 #${step.knowledgeId}`)
  ))
}

export function getWorkflowInstructionsForReadOnly(answer, stepCount) {
  return getWorkflowInstructionsEditorValue(answer, stepCount).trim()
}
