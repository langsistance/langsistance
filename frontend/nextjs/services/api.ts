import { getValidToken } from '@/lib/auth-client'
import { assertApiResponseSuccess } from '@/lib/apiResponse'
import { withWebKnowledgePushFilter } from '@/lib/webKnowledgeQueries'

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE || 'https://api.copiioai.com'

async function authHeaders(): Promise<Record<string, string>> {
  if (typeof window === 'undefined') throw new Error('API service is client-only')
  const token = await getValidToken()
  if (!token) throw new Error('Not authenticated')
  return { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const headers = await authHeaders()
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const errorBody = await res.text().catch(() => '')
    throw new Error(`${path} failed: ${res.status}${errorBody ? ` — ${errorBody}` : ''}`)
  }
  return assertApiResponseSuccess(await res.json(), `${path} failed`) as T
}

async function get<T>(path: string, params: Record<string, string | number> = {}): Promise<T> {
  const headers = await authHeaders()
  const entries = Object.entries(params).map(([k, v]) => [k, String(v)] as [string, string])
  const qs = new URLSearchParams(entries).toString()
  const url = `${BASE_URL}${path}${qs ? '?' + qs : ''}`
  const res = await fetch(url, { headers })
  if (!res.ok) {
    const errorBody = await res.text().catch(() => '')
    throw new Error(`${path} failed: ${res.status}${errorBody ? ` — ${errorBody}` : ''}`)
  }
  return assertApiResponseSuccess(await res.json(), `${path} failed`) as T
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ApiResult = Promise<any>

export const queryTools = (params: Record<string, string | number> = {}): ApiResult => {
  const { page = 1, limit = 100, ...rest } = params
  return get('/query_tools', { ...rest, offset: Number(page) - 1, limit })
}
export const queryToolById = (toolId: number | string): ApiResult => get('/query_tool_by_id', { tool_id: toolId })
export const createToolFromCustom = (body: unknown): ApiResult => post('/create_tool_from_custom', body)
export const createToolFromOpenapi = (body: unknown): ApiResult => post('/create_tool_from_openapi', body)
export const updateTool = (body: unknown): ApiResult => post('/update_tool', body)
export const deleteTool = (body: unknown): ApiResult => post('/delete_tool', body)

export const queryKnowledge = ({ search = '', page = 1, limit = 10 }: { search?: string; page?: number; limit?: number } = {}): ApiResult =>
  get('/query_knowledge', withWebKnowledgePushFilter({ query: search, offset: page - 1, limit }))
export const createKnowledge = (body: unknown): ApiResult => post('/create_knowledge', body)
export const updateKnowledge = (body: unknown): ApiResult => post('/update_knowledge', body)
export const deleteKnowledge = (body: unknown): ApiResult => post('/delete_knowledge', body)

export const queryPublicKnowledge = ({ search = '', page = 1, limit = 10 }: { search?: string; page?: number; limit?: number } = {}): ApiResult =>
  get('/query_public_knowledge', withWebKnowledgePushFilter({ query: search, offset: page - 1, limit }))
export const copyKnowledge = (body: unknown): ApiResult => post('/copy_knowledge', body)

export const authorizeKnowledgeAccess = (body: unknown): ApiResult => post('/authorize_knowledge_access', body)
export const handleKnowledgeShare = (body: unknown): ApiResult => post('/handle_knowledge_share', body)
export const queryKnowledgeShares = (params: Record<string, string | number>): ApiResult => get('/query_knowledge_shares', withWebKnowledgePushFilter(params))
export const getUserSharedKnowledge = (params: Record<string, string | number>): ApiResult => get('/get_user_shared_knowledge', withWebKnowledgePushFilter(params))
export const cancelKnowledgeShare = (body: unknown): ApiResult => post('/cancel_knowledge_share', body)

// ── Scene API ────────────────────────────────────────────────────────────

export const getAvailableScenes = (): ApiResult => get('/scenes/available')
export const getSceneKnowledge = (sceneId: number): ApiResult =>
  get(`/scenes/${sceneId}/knowledge`)
export const getUserScenes = (): ApiResult => get('/user/scenes')
export const getUserSceneStatus = (): ApiResult => get('/user/scenes/status')
export const updateUserScenes = (sceneIds: number[]): ApiResult =>
  post('/user/scenes', { scene_ids: sceneIds })
export const markOnboarded = (): ApiResult => post('/user/onboarded', {})

// ── Feedback & Messages ─────────────────────────────────────────────────

export const submitFeedback = (content: string): ApiResult => post('/submit_feedback', { content })
export const getMessages = (): ApiResult => get('/messages')
export const getUnreadCount = (): ApiResult => get('/messages/unread_count')
export const markMessageRead = (messageId: number): ApiResult => post(`/messages/${messageId}/read`, {})
export const markAllMessagesRead = (): ApiResult => post('/messages/read_all', {})

// ── Chat Stream ─────────────────────────────────────────────────────────

export async function queryStream(
  query: string,
  queryId: string,
  abortSignal: AbortSignal,
  conversationHistory: { role: string; content: string }[] = [],
): Promise<ReadableStream<Uint8Array>> {
  const headers = await authHeaders()
  const res = await fetch(`${BASE_URL}/query_stream`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ query, query_id: queryId, push_filter: 2, conversation_history: conversationHistory }),
    signal: abortSignal,
  })
  if (!res.ok) throw new Error(`/query_stream failed: ${res.status}`)
  if (!res.body) throw new Error('/query_stream: response body is null')
  return res.body
}

export async function queryStreamWithFiles(
  query: string,
  queryId: string,
  abortSignal: AbortSignal,
  files: File[],
  conversationHistory: { role: string; content: string }[] = [],
): Promise<ReadableStream<Uint8Array>> {
  const formData = new FormData()
  formData.append('query', query)
  formData.append('query_id', queryId)
  formData.append('push_filter', '2')
  formData.append('conversation_history', JSON.stringify(conversationHistory))
  for (const file of files) {
    formData.append('patent_files', file)
  }
  const headers = await authHeaders()
  // Remove Content-Type so browser auto-sets multipart boundary.
  // Delete both casings — authHeaders returns 'Content-Type' (Pascal case).
  delete headers['Content-Type']
  delete headers['content-type']
  const res = await fetch(`${BASE_URL}/query_stream`, {
    method: 'POST',
    headers: headers,
    body: formData,
    signal: abortSignal,
  })
  if (!res.ok) throw new Error(`/query_stream (multipart) failed: ${res.status}`)
  if (!res.body) throw new Error('/query_stream: response body is null')
  return res.body
}

// ── Sessions ──────────────────────────────────────────────────────────────

export interface SessionItem {
  session_id: string
  title: string
  status: number
  long_task_ids: string[] | null
  create_time: string | null
  update_time: string | null
}

export async function getSessions(): Promise<SessionItem[]> {
  const headers = await authHeaders()
  const res = await fetch(`${BASE_URL}/sessions`, { headers })
  const data = await res.json()
  assertApiResponseSuccess(data)
  return (data.sessions || []) as SessionItem[]
}

// ── Long Task ─────────────────────────────────────────────────────────────

export async function pollLongTaskStatus(taskId: string): Promise<{
  success: boolean
  status: string
  progress?: number
  current_phase?: string
  current_step?: string
  report_files?: { format: string; filename: string; size: number }[]
  result_summary?: string
  error_message?: string
}> {
  return get(`/long_task/${taskId}/status`)
}

export function getLongTaskReportUrl(taskId: string, format: 'pdf' | 'docx' = 'pdf'): string {
  return `${BASE_URL}/long_task/${taskId}/report?format=${format}`
}
