export interface ToolWithPush {
  push?: number
}

export function filterKnowledgeBaseTools<T extends ToolWithPush>(tools: T[]): T[] {
  return tools.filter((tool) => tool.push === 2)
}
