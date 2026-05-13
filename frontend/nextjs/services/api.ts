import { getIdToken } from 'firebase/auth'
import { auth } from '@/lib/firebase'

const BASE_URL = 'https://api.copiioai.com'

async function authHeaders(): Promise<Record<string, string>> {
  if (typeof window === 'undefined') throw new Error('API service is client-only')
  const user = auth.currentUser
  if (!user) throw new Error('Not authenticated')
  const token = await getIdToken(user)
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
  return res.json()
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
  return res.json()
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ApiResult = Promise<any>

export const queryTools = (params: Record<string, string | number> = {}): ApiResult => {
  const { page = 1, limit = 100, ...rest } = params
  return get('/query_tools', { ...rest, offset: Number(page) - 1, limit })
}
export const createToolFromCustom = (body: unknown): ApiResult => post('/create_tool_from_custom', body)
export const createToolFromOpenapi = (body: unknown): ApiResult => post('/create_tool_from_openapi', body)
export const updateTool = (body: unknown): ApiResult => post('/update_tool', body)
export const deleteTool = (body: unknown): ApiResult => post('/delete_tool', body)

export const queryKnowledge = ({ search = '', page = 1, limit = 10 }: { search?: string; page?: number; limit?: number } = {}): ApiResult =>
  get('/query_knowledge', { query: search, offset: page - 1, limit })
export const createKnowledge = (body: unknown): ApiResult => post('/create_knowledge', body)
export const updateKnowledge = (body: unknown): ApiResult => post('/update_knowledge', body)
export const deleteKnowledge = (body: unknown): ApiResult => post('/delete_knowledge', body)

export const queryPublicKnowledge = ({ search = '', page = 1, limit = 10 }: { search?: string; page?: number; limit?: number } = {}): ApiResult =>
  get('/query_public_knowledge', { query: search, offset: page - 1, limit })
export const copyKnowledge = (body: unknown): ApiResult => post('/copy_knowledge', body)

export const authorizeKnowledgeAccess = (body: unknown): ApiResult => post('/authorize_knowledge_access', body)
export const handleKnowledgeShare = (body: unknown): ApiResult => post('/handle_knowledge_share', body)
export const queryKnowledgeShares = (params: Record<string, string | number>): ApiResult => get('/query_knowledge_shares', params)
export const getUserSharedKnowledge = (params: Record<string, string | number>): ApiResult => get('/get_user_shared_knowledge', params)
export const cancelKnowledgeShare = (body: unknown): ApiResult => post('/cancel_knowledge_share', body)

export async function queryStream(query: string, queryId: string, abortSignal: AbortSignal): Promise<ReadableStream<Uint8Array>> {
  const headers = await authHeaders()
  const res = await fetch(`${BASE_URL}/query_stream`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ query, query_id: queryId, push_filter: 2 }),
    signal: abortSignal,
  })
  if (!res.ok) throw new Error(`/query_stream failed: ${res.status}`)
  if (!res.body) throw new Error('/query_stream: response body is null')
  return res.body
}
