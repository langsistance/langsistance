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

// Sessions
export const listSessions = (userId) =>
  get('/sessions', { user_id: userId })

export const getSession = (sessionId) =>
  get(`/session/${sessionId}`)

export const createSession = (body) =>
  post('/session', body)

export const appendMessage = (sessionId, body) =>
  post(`/session/${sessionId}/message`, body)

export const archiveSession = (sessionId) =>
  fetch(`${BASE_URL}/session/${sessionId}`, {
    method: 'DELETE',
    headers: awaitAuthHeaders(),
  }).then((r) => {
    if (!r.ok) throw new Error(`DELETE /session/${sessionId} failed: ${r.status}`)
    return r.json()
  })

// Long Task
export const pollTaskStatus = (taskId) =>
  get(`/long_task/${taskId}/status`)

export const getReportDownloadUrl = (taskId, format = 'pdf') =>
  `${BASE_URL}/long_task/${taskId}/report?format=${format}`

// Helper for DELETE that needs auth headers but no body
async function awaitAuthHeaders() {
  const user = auth.currentUser
  if (!user) throw new Error('Not authenticated')
  const token = await getIdToken(user)
  return { Authorization: `Bearer ${token}` }
}
