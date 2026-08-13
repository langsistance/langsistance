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
  results, activeRowId, onSelect, onOpenTab,
}: {
  results: ResultsPayload
  activeRowId: string | null
  onSelect: (model: RowModel) => void
  onOpenTab: (model: RowModel, tab: string) => void
}) {
  const { t } = useI18n()
  const [sort, setSort] = useState<'relevance' | 'date' | 'assignee'>('relevance')
  const [sourceFilter, setSourceFilter] = useState<string>('all')

  const models = useMemo<RowModel[]>(() => {
    const dateCol = findRoleColumn(results.columns, 'publication_date')
    const assigneeCol = findRoleColumn(results.columns, 'assignee')
    const list = results.rows.map((row) => buildRowModel(row, results.columns, results.source) as RowModel)
    if (sort === 'date' && dateCol) {
      const key = dateCol.label || dateCol.key
      list.sort((a, b) =>
        String(b.fields.find(([k]) => k === key)?.[1] || '').localeCompare(
          String(a.fields.find(([k]) => k === key)?.[1] || ''),
        ),
      )
    } else if (sort === 'assignee' && assigneeCol) {
      const key = assigneeCol.label || assigneeCol.key
      list.sort((a, b) =>
        (a.meta.find((m) => m.label === key)?.value ?? '').localeCompare(
          b.meta.find((m) => m.label === key)?.value ?? '',
        ),
      )
    }
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
          />
        ))}
      </div>
    </div>
  )
}
