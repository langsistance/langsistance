'use client'

import { useMemo, useState } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { buildRowModel, findRoleColumn } from '@/lib/results'
import ResultRow from './ResultRow'

interface ResultsPayload {
  setId: string
  source: string
  columns: Array<{ key: string; label: string; role: string }>
  rows: Array<Record<string, unknown>>
}

interface RowModel {
  id: string
  title: string
  meta: Array<{ label: string; value: string }>
  patentId: string
  applicationNumber: string
  url: string
  source: string
  isDocument: boolean
  fields: Array<[string, string]>
}

export default function ResultList({
  results, activeRowId, queryText, onSelect, onOpenTab, onProsecution, onCollapse,
}: {
  results: ResultsPayload
  activeRowId: string | null
  queryText?: string
  onSelect: (model: RowModel, index: number) => void
  onOpenTab: (model: RowModel, index: number, tab: string) => void
  onProsecution: (model: RowModel) => void
  onCollapse?: () => void
}) {
  const { t } = useI18n()
  const [sort, setSort] = useState<'relevance' | 'date' | 'assignee'>('relevance')
  const [sourceFilter, setSourceFilter] = useState<string>('all')

  const models = useMemo<Array<{ model: RowModel; index: number }>>(() => {
    const dateCol = findRoleColumn(results.columns, 'publication_date')
    const assigneeCol = findRoleColumn(results.columns, 'assignee')
    const entries = results.rows.map((row, index) => {
      const model = buildRowModel(row, results.columns, results.source) as RowModel
      return { row, model, index }
    })
    if (sort === 'date' && dateCol) {
      // Compare the raw column value off the row, not the rendered label, so
      // rows whose date is absent sort deterministically.
      entries.sort((a, b) =>
        String(b.row[dateCol.key] || '').localeCompare(
          String(a.row[dateCol.key] || ''),
        ),
      )
    } else if (sort === 'assignee' && assigneeCol) {
      const key = assigneeCol.label || assigneeCol.key
      entries.sort((a, b) =>
        (a.model.meta.find((m) => m.label === key)?.value ?? '').localeCompare(
          b.model.meta.find((m) => m.label === key)?.value ?? '',
        ),
      )
    }
    const list = entries.map((e) => ({ model: e.model, index: e.index }))
    if (sourceFilter !== 'all') {
      return list.filter(({ model }) => model.source === sourceFilter)
    }
    return list
  }, [results, sort, sourceFilter])

  const sources = useMemo(() => {
    const set = new Set(models.map(({ model }) => model.source))
    return Array.from(set)
  }, [models])

  return (
    <div className="results-list">
      <div className="results-list-toolbar">
        {onCollapse && (
          <button
            className="results-collapse-btn"
            onClick={onCollapse}
            aria-label={t('results.collapseList')}
            title={t('results.collapseList')}
          >
            ⟨
          </button>
        )}
        {queryText && (
          <span className="results-list-query" title={queryText}>{queryText}</span>
        )}
        <span className="results-list-count">{t('results.selectedCount').replace('{count}', String(models.length))}</span>
        <select value={sort} onChange={(e) => setSort(e.target.value as any)} aria-label="sort">
          <option value="relevance">{t('results.sortRelevance')}</option>
          <option value="date">{t('results.sortDate')}</option>
          <option value="assignee">{t('results.sortAssignee')}</option>
        </select>
        <select value={sourceFilter} onChange={(e) => setSourceFilter(e.target.value)} aria-label="source filter">
          <option value="all">{t('results.filterAll')}</option>
          {sources.map((source) => (
            <option key={source} value={source}>{source}</option>
          ))}
        </select>
      </div>
      <div className="results-list-scroll">
        {models.map(({ model, index }) => (
          <ResultRow
            // The index keeps keys unique — duplicate id/title pairs across
            // rows (document lists share descriptions) once leaked stale DOM
            // nodes into the list and broke selection on set switch.
            key={`${index}-${model.id}`}
            model={model}
            index={index}
            active={activeRowId === String(index)}
            onSelect={onSelect}
            onOpenTab={onOpenTab}
            onProsecution={onProsecution}
          />
        ))}
      </div>
    </div>
  )
}
