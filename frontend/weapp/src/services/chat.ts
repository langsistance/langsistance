import { request } from './api'

/** 会话消息（与后端 conversations.messages 结构一致）。 */
export interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
  [key: string]: any
}

export interface SessionDetail {
  session_id: string
  title: string
  messages: ChatMsg[]
  update_time?: string | null
}

/** 建会话：首条提问时调用一次（scene 1 = 专利检索默认场景）。 */
export async function createSession(
  title: string,
  messages: ChatMsg[],
): Promise<string> {
  const body = await request<{ success: boolean; session_id: string }>(
    '/session',
    {
      method: 'POST',
      data: { scene_id: 1, title: title.slice(0, 60), messages },
    },
  )
  return body.session_id
}

/** 读会话历史（GET /session/{id}，服务端未要求鉴权，仍带 token）。 */
export async function fetchSession(sessionId: string): Promise<SessionDetail> {
  const body = await request<{
    success: boolean
    session: SessionDetail
  }>(`/session/${sessionId}`)
  return body.session
}

/** 保存整段会话消息（每轮结束调用，web 同款持久化）。 */
export async function saveMessages(
  sessionId: string,
  messages: ChatMsg[],
  title = '',
): Promise<void> {
  await request(`/session/${sessionId}/messages`, {
    method: 'PUT',
    data: { messages, title: title.slice(0, 60) },
  })
}

/** 从助手终稿文本抽取专利号候选（零后端依赖的过渡面板；
 *  完整结果面板随结果事件通道 M3 落地）。 */
const PATENT_RE = /(CN\d{7,12}[A-Z]?|\b\d{8}\b)/g

export function extractPatentIds(text: string): string[] {
  if (!text) return []
  const seen = new Set<string>()
  for (const m of text.matchAll(PATENT_RE)) {
    seen.add(m[1])
  }
  return Array.from(seen).slice(0, 20)
}
