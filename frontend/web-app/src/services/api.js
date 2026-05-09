import { getIdToken } from 'firebase/auth'
import { auth } from '../firebase'

const BASE_URL = 'https://api.copiioai.com'

async function authHeaders() {
  const user = auth.currentUser
  if (!user) throw new Error('Not authenticated')
  const token = await getIdToken(user)
  return { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }
}

async function post(path, body) {
  const headers = await authHeaders()
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`)
  return res.json()
}

async function get(path, params = {}) {
  const headers = await authHeaders()
  const qs = new URLSearchParams(params).toString()
  const url = `${BASE_URL}${path}${qs ? '?' + qs : ''}`
  const res = await fetch(url, { headers })
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`)
  return res.json()
}

// Tools
export const queryTools = (params) => get('/query_tools', params)
export const createToolFromCustom = (body) => post('/create_tool_from_custom', body)
export const createToolFromOpenapi = (body) => post('/create_tool_from_openapi', body)
export const updateTool = (body) => post('/update_tool', body)
export const deleteTool = (body) => post('/delete_tool', body)

// Knowledge
export const queryKnowledge = (params) => get('/query_knowledge', params)
export const createKnowledge = (body) => post('/create_knowledge', body)
export const updateKnowledge = (body) => post('/update_knowledge', body)
export const deleteKnowledge = (body) => post('/delete_knowledge', body)

// Public knowledge
export const queryPublicKnowledge = (params) => get('/query_public_knowledge', params)
export const copyKnowledge = (body) => post('/copy_knowledge', body)

// Sharing
export const authorizeKnowledgeAccess = (body) => post('/authorize_knowledge_access', body)
export const handleKnowledgeShare = (body) => post('/handle_knowledge_share', body)
export const queryKnowledgeShares = (params) => get('/query_knowledge_shares', params)
export const getUserSharedKnowledge = (params) => get('/get_user_shared_knowledge', params)
export const cancelKnowledgeShare = (body) => post('/cancel_knowledge_share', body)

// Chat streaming — returns response body as ReadableStream
export async function queryStream(query, queryId, abortSignal) {
  const headers = await authHeaders()
  const res = await fetch(`${BASE_URL}/query_stream`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ query, query_id: queryId, push_filter: 2 }),
    signal: abortSignal,
  })
  if (!res.ok) throw new Error(`/query_stream failed: ${res.status}`)
  return res.body
}
