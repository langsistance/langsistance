export function getKnowledgeWorkflowAnswer(stepCount, lang = 'en') {
  const count = Number(stepCount)
  if (lang === 'zh') {
    return `组合知识：按顺序执行 ${count} 个知识步骤。`
  }
  return `Composed knowledge: execute ${count} knowledge steps in order.`
}
