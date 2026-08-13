/**
 * Browser-local persistence for patent search result sets.
 * Result sets are pruned copies of the decoded format=json artifact,
 * kept in localStorage under one key.  All storage access is guarded —
 * unavailable storage (private mode, disabled site data) degrades to
 * the in-memory behaviour silently.
 */

import { pruneResultsForPersistence } from './results.js'

export const RESULTS_STORE_KEY = 'copiioai_results'
export const MAX_RESULT_SETS = 100
export const MAX_INDEX_ENTRIES = 200

export function emptyResultsStore() {
  return { sets: {}, index: [] }
}

export function loadResultsStore(storage) {
  if (!storage) return emptyResultsStore()
  try {
    const parsed = JSON.parse(storage.getItem(RESULTS_STORE_KEY))
    if (
      !parsed
      || typeof parsed !== 'object'
      || typeof parsed.sets !== 'object'
      || parsed.sets === null
      || !Array.isArray(parsed.index)
    ) {
      return emptyResultsStore()
    }
    return parsed
  } catch {
    return emptyResultsStore()
  }
}

function dropOldestSet(store) {
  const keys = Object.keys(store.sets)
  if (keys.length === 0) return store
  const { [keys[0]]: _dropped, ...rest } = store.sets
  return { ...store, sets: rest }
}

export function persistResultsSet(store, results, meta) {
  const pruned = pruneResultsForPersistence(results)
  let sets = { ...(store?.sets || {}), [results.setId]: pruned }
  const keys = Object.keys(sets)
  while (keys.length > MAX_RESULT_SETS) {
    delete sets[keys[0]]
    keys.shift()
  }
  const index = [
    {
      setId: results.setId,
      sessionId: meta.sessionId ?? null,
      queryText: meta.queryText ?? '',
      savedAt: meta.savedAt ?? 0,
    },
    ...(store?.index || []),
  ].slice(0, MAX_INDEX_ENTRIES)
  return { sets, index }
}

export function saveResultsStore(storage, store) {
  if (!storage) return
  let current = store
  while (true) {
    try {
      storage.setItem(RESULTS_STORE_KEY, JSON.stringify(current))
      return
    } catch (error) {
      // Only quota pressure justifies dropping sets; permanent errors
      // (blocked site data, serialization failure) must not discard data.
      const name = error && error.name ? error.name : ''
      const code = error && error.code ? error.code : 0
      const isQuota = name === 'QuotaExceededError' || code === 22 || code === 1014
      if (!isQuota) return
      const next = dropOldestSet(current)
      if (Object.keys(next.sets).length === Object.keys(current.sets).length) {
        return // 没有可丢的了——静默放弃
      }
      current = next
    }
  }
}

export function persistResultsSetToStorage(storage, results, meta) {
  saveResultsStore(storage, persistResultsSet(loadResultsStore(storage), results, meta))
}

export function restoreResultsInMessages(messages, store) {
  if (!Array.isArray(messages) || !store || !Array.isArray(store.index)) return messages
  const sets = store.sets || {}
  return messages.map((msg, index) => {
    if (msg.role !== 'assistant' || msg.results || index === 0) return msg
    const previous = messages[index - 1]
    if (!previous || previous.role !== 'user') return msg
    for (const entry of store.index) {
      if (entry.queryText && entry.queryText === previous.content && sets[entry.setId]) {
        return { ...msg, results: { setId: entry.setId, ...sets[entry.setId] } }
      }
    }
    return msg
  })
}

export function buildStoredMessage(setId, store) {
  if (!store || !setId) return null
  const set = store.sets && store.sets[setId]
  if (!set) return null
  return {
    id: `stored-${setId}`,
    role: 'assistant',
    content: '',
    results: { setId, ...set },
  }
}
