export function getKnowledgeWorkflowAnswer(stepCount, lang = 'en') {
  const count = Number(stepCount)
  if (lang === 'zh') {
    return `组合知识：按顺序执行 ${count} 个知识步骤。`
  }
  return `Composed knowledge: execute ${count} knowledge steps in order.`
}

export function getWorkflowInstructionsForSave(value, stepCount, lang = 'en') {
  const instructions = String(value || '').trim()
  if (!instructions || isGeneratedWorkflowAnswer(instructions)) {
    return getKnowledgeWorkflowAnswer(stepCount, lang)
  }
  return instructions
}

export function isGeneratedWorkflowAnswer(value, stepCount) {
  const answer = String(value || '').trim()
  if (!answer) return false

  if (stepCount != null) {
    if (
      answer === getKnowledgeWorkflowAnswer(stepCount, 'en') ||
      answer === getKnowledgeWorkflowAnswer(stepCount, 'zh')
    ) {
      return true
    }
  }

  return (
    /^Composed knowledge: execute \d+ knowledge steps in order\.$/.test(answer) ||
    /^组合知识：按顺序执行 \d+ 个知识步骤。$/.test(answer)
  )
}

export function getWorkflowInstructionsEditorValue(value, stepCount) {
  const answer = String(value || '')
  return isGeneratedWorkflowAnswer(answer, stepCount) ? '' : answer
}
