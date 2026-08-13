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
  results, activeRowId, onSelect, onOpenTab, onProsecution,
}: {
  results: ResultsPayload
  activeRowId: string | null
  onSelect: (model: RowModel) => void
  onOpenTab: (model: RowModel, tab: string) => void
  onProsecution: (model: RowModel) => void
}) {
  const { t } = useI18n()
  const [sort, setSort] = useState<'relevance' | 'date' | 'assignee'>('relevance')
  const [sourceFilter, setSourceFilter] = useState<string>('all')

  const models = useMemo<RowModel[]>(() => {
    const dateCol = findRoleColumn(results.columns, 'publication_date')
    const assigneeCol = findRoleColumn(results.columns, 'assignee')
    const entries = results.rows.map((row, index) => {
      const model = buildRowModel(row, results.columns, results.source) as RowModel
      // Fallback id collision — rows with no usable title/patent id all get
      // 'row'; disambiguate with the original index.
      if (model.id === 'row') model.id = `row-${index}`
      return { row, model }
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
    const list = entries.map((e) => e.model)
    if (sourceFilter !== 'all') {
      return list.filter((m) => m.source === sourceFilter)
    }
    return list
  }, [results, sort, sourceFilter])

  const sources = useMemo(() => {
    const set = new Set(models.map((m) => m.source))
    return Array.from(set)
  }, [models])

  return (
    <div className="results-list">
      <div className="results-list-toolbar">
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
        {models.map((model) => (
          <ResultRow
            key={model.id + model.title}
            model={model}
            active={activeRowId === model.id}
            onSelect={onSelect}
            onOpenTab={onOpenTab}
            onProsecution={onProsecution}
          />
        ))}
      </div>
    </div>
  )
}
