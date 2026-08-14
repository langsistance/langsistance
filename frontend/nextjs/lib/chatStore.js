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
        // Artifact chunks are base64-heavy; downloads are not restored
        // across provider remounts (same trade-off as session loads).
        artifacts: [],
      }
      if (msg.taskId) pruned.taskId = msg.taskId
      if (msg.resultSummary) pruned.resultSummary = String(msg.resultSummary).slice(0, MAX_PERSIST_SUMMARY_CHARS)
      if (Array.isArray(msg.patent_ids) && msg.patent_ids.length > 0) pruned.patent_ids = msg.patent_ids
      if (msg.results) pruned.results = pruneResultsForPersistence(msg.results)
      return pruned
    })
    .filter(Boolean)
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
  } catch {
    // Quota / blocked site data — the conversation stays in memory only.
  }
}

export function persistChatToStorage(storage, messages) {
  saveChatStore(storage, pruneMessagesForPersistence(messages))
}
