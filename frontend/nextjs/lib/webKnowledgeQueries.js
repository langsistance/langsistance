export const WEB_KNOWLEDGE_PUSH_FILTER = 2

export function withWebKnowledgePushFilter(params = {}) {
  return {
    ...params,
    push_filter: WEB_KNOWLEDGE_PUSH_FILTER,
  }
}
