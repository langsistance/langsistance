import { request } from './api'

export interface SessionItem {
  session_id: string
  title: string
  status?: number
  long_task_ids?: string[]
  create_time?: string | null
  update_time?: string | null
}

/** GET /sessions —— 当前用户会话列表（按 update_time 倒序，后端已排）。 */
export async function fetchSessions(): Promise<SessionItem[]> {
  const body = await request<{ success: boolean; sessions: SessionItem[] }>(
    '/sessions',
  )
  return body.sessions || []
}
