/**
 * Browser-local persistence for the in-progress conversation.
 *
 * The landing page (/) and the /app/* routes each mount their own
 * ChatProvider, so a client-side navigation between them destroys the
 * React state holding the conversation (question + answer).  Persisting a
 * pruned copy to sessionStorage and hydrating on mount bridges the two
 * providers.  sessionStorage (not localStorage) keeps a fresh tab a fresh
 * conversation, while still surviving refreshes and full-page fallbacks.
 * All storage access is guarded — unavailable storage (private mode,
 * disabled site data) degrades to the in-memory behaviour silently.
 */

import { pruneResultsForPersistence } from './results.js'

export const CHAT_STORE_KEY = 'copiioai_chat'
// Long-task report previews are the one unbounded field copied through;
// a cap keeps a single huge report from exhausting the storage quota and
// silently dropping the whole conversation.
export const MAX_PERSIST_SUMMARY_CHARS = 10000

export function pruneMessagesForPersistence(messages) {
  if (!Array.isArray(messages)) return []
  return messages
    .map((msg) => {
      if (!msg || typeof msg.content !== 'string') return null
      const pruned = {
        id: msg.id,
        role: msg.role,
        content: msg.content,
        // Artifact chunks are preserved so the Excel/CSV download buttons
        // keep working after a provider remount.  The json artifact's
        // chunks are dropped — its data already lives in msg.results and
        // the results store, and its chunks are the largest payload
        // (repeated rows) — keeping only csv/xlsx chunks fits the
        // sessionStorage quota.  saveChatStore strips the rest only when
        // the quota still forces it.
        artifacts: (Array.isArray(msg.artifacts) ? msg.artifacts : []).map((artifact) => {
          if (artifact && artifact.format === 'json') {
            return { ...artifact, chunks: [] }
          }
          return artifact
        }),
      }
      if (msg.taskId) pruned.taskId = msg.taskId
      if (msg.resultSummary) pruned.resultSummary = String(msg.resultSummary).slice(0, MAX_PERSIST_SUMMARY_CHARS)
      if (Array.isArray(msg.patent_ids) && msg.patent_ids.length > 0) pruned.patent_ids = msg.patent_ids
      // ReAct step timeline is small (bounded text per step) — keep it so
      // the collapsed "elapsed · N steps" header survives remounts.
      if (Array.isArray(msg.agentSteps) && msg.agentSteps.length > 0) pruned.agentSteps = msg.agentSteps
      if (Number.isFinite(msg.elapsedSeconds)) pruned.elapsedSeconds = msg.elapsedSeconds
      if (msg.results) pruned.results = pruneResultsForPersistence(msg.results)
      return pruned
    })
    .filter(Boolean)
}

export function stripArtifactChunks(messages) {
  if (!Array.isArray(messages)) return []
  return messages.map((msg) =>
    Array.isArray(msg && msg.artifacts) ? { ...msg, artifacts: [] } : msg
  )
}

export function loadChatStore(storage) {
  if (!storage) return []
  try {
    const parsed = JSON.parse(storage.getItem(CHAT_STORE_KEY))
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function saveChatStore(storage, messages) {
  if (!storage) return
  try {
    storage.setItem(CHAT_STORE_KEY, JSON.stringify(messages))
    return
  } catch (error) {
    const name = error && error.name ? error.name : ''
    const code = error && error.code ? error.code : 0
    const isQuota = name === 'QuotaExceededError' || code === 22 || code === 1014
    if (!isQuota) return
  }
  // Quota pressure — retry once with artifact chunks stripped; the
  // conversation survives, only the download payloads degrade.
  try {
    storage.setItem(CHAT_STORE_KEY, JSON.stringify(stripArtifactChunks(messages)))
  } catch {
    // Storage unavailable — the conversation stays in memory only.
  }
}

export function persistChatToStorage(storage, messages) {
  saveChatStore(storage, pruneMessagesForPersistence(messages))
}

/**
 * Last backend session per browser user (需求 2: 会话持久化).
 *
 * Pure-chat conversations now get a backend session_id too; this record
 * lets a fresh mount (no session_id in the URL) restore the most recent
 * conversation.  The uid is stored alongside so a different account on
 * the same browser never resurrects another user's conversation.
 */
export const LAST_SESSION_KEY = 'copiioai_last_session'

export function saveLastSession(storage, sessionId, uid) {
  if (!storage || !sessionId) return
  try {
    storage.setItem(LAST_SESSION_KEY, JSON.stringify({ sid: sessionId, uid: uid || null }))
  } catch {
    // Storage unavailable — degrade silently
  }
}

export function loadLastSession(storage) {
  if (!storage) return null
  try {
    const raw = JSON.parse(storage.getItem(LAST_SESSION_KEY))
    if (!raw || typeof raw !== 'object' || typeof raw.sid !== 'string') return null
    return { sid: raw.sid, uid: typeof raw.uid === 'string' ? raw.uid : null }
  } catch {
    return null
  }
}

export function clearLastSession(storage) {
  if (!storage) return
  try {
    storage.removeItem(LAST_SESSION_KEY)
  } catch {
    // Ignore
  }
}
