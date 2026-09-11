import { request } from './api'

export interface SessionItem {
  session_id: string
  title: string
  status?: number
  long_task_ids?: string[]
  create_time?: string | null
  update_time?: string | null
}

/** 会话标题客户端上限（与后端 256 的列宽守卫不同，这是产品口径）。 */
export const MAX_TITLE_LEN = 60

/** GET /sessions —— 当前用户会话列表（按 update_time 倒序，后端已排）。 */
export async function fetchSessions(): Promise<SessionItem[]> {
  const body = await request<{ success: boolean; sessions: SessionItem[] }>(
    '/sessions',
  )
  return body.sessions || []
}

/** 重命名会话（专用端点，只改标题，不重写 messages）。 */
export async function renameSession(
  sessionId: string,
  title: string,
): Promise<void> {
  await request(`/session/${sessionId}/title`, {
    method: 'PUT',
    data: { title: title.slice(0, MAX_TITLE_LEN) },
  })
}

/** 归档会话（后端置 status=2，列表与详情都不再返回）。 */
export async function archiveSession(sessionId: string): Promise<void> {
  await request(`/session/${sessionId}`, { method: 'DELETE' })
}
