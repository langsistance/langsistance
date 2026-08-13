/**
 * Results payload helpers shared by the chat result card and the
 * split-view results page.  Works purely on the structured payload
 * decoded from the format=json artifact.
 */

import { buildStoredMessage } from './resultsStore.js'

export function findRoleColumn(columns, role) {
  const list = Array.isArray(columns) ? columns : []
  return list.find((col) => col && col.role === role) || null
}

export function columnValue(row, column) {
  if (!row || !column) return ''
  const value = row[column.key]
  return value === undefined || value === null ? '' : String(value)
}

const META_ROLES = ['patent_id', 'application_number', 'publication_number', 'assignee', 'publication_date']

export function buildRowModel(row, columns, source) {
  const list = Array.isArray(columns) ? columns : []
  const docTitleCol = findRoleColumn(list, 'document_title')
  const isDocument = Boolean(docTitleCol)

  const titleCol = isDocument ? docTitleCol : findRoleColumn(list, 'title')
  const title = columnValue(row, titleCol)

  const meta = []
  for (const role of META_ROLES) {
    const col = findRoleColumn(list, role)
    if (!col) continue
    const value = columnValue(row, col)
    if (value) meta.push({ label: col.label || col.key, value })
  }

  const fields = []
  for (const col of list) {
    const value = columnValue(row, col)
    if (!value) continue
    if (isDocument && col.role === 'document_title') continue
    fields.push([col.label || col.key, value])
  }

  const patentIdCol = findRoleColumn(list, 'patent_id')
  const appNumCol = findRoleColumn(list, 'application_number')
  const urlCol = findRoleColumn(list, 'url')

  return {
    id: String(columnValue(row, patentIdCol) || title || fields[0]?.[1] || 'row'),
    title,
    meta,
    patentId: columnValue(row, patentIdCol),
    applicationNumber: columnValue(row, appNumCol),
    url: columnValue(row, urlCol),
    source,
    isDocument,
    fields,
  }
}

export const MAX_PERSIST_ROWS = 50
export const MAX_PERSIST_ABSTRACT_CHARS = 500

export function pruneResultsForPersistence(
  results,
  { maxRows = MAX_PERSIST_ROWS, abstractLimit = MAX_PERSIST_ABSTRACT_CHARS } = {},
) {
  if (!results || !Array.isArray(results.rows)) return results
  const columns = Array.isArray(results.columns) ? results.columns : []
  const abstractCols = columns.filter((col) => col && col.role === 'abstract')
  const displayCols = columns.filter((col) => col && col.role !== 'text')
  const displayKeys = new Set(displayCols.map((col) => col.key))

  const rows = results.rows.slice(0, maxRows).map((row) => {
    const pruned = {}
    for (const key of Object.keys(row)) {
      if (!displayKeys.has(key)) continue
      let value = row[key]
      if (abstractCols.some((col) => col.key === key) && typeof value === 'string') {
        value = value.slice(0, abstractLimit)
      }
      pruned[key] = value
    }
    return pruned
  })

  return {
    setId: results.setId,
    source: results.source,
    columns: displayCols,
    rows,
  }
}

/**
 * Resolve the message whose results the results page should display.
 * Prefers the URL's setId; when nothing matches in memory — the
 * auto-navigation state race right after streaming, or a stale set in
 * the URL — falls back to the newest message carrying results so the
 * list still renders.  When the in-memory match comes up empty, an
 * optional `store` may supply a synthetic message restored from
 * localStorage for the setId (Task 2), before the newest in-memory
 * message is used as the last resort.
 */
export function resolveActiveResultsMessage(messages, urlSetId, store) {
  const list = Array.isArray(messages) ? messages : []
  let newest = null
  for (const message of list) {
    if (!message || !message.results) continue
    if (urlSetId && message.results.setId === urlSetId) return message
    newest = message
  }
  if (urlSetId && store) {
    const storedMessage = buildStoredMessage(urlSetId, store)
    if (storedMessage) return storedMessage
  }
  return newest || undefined
}
