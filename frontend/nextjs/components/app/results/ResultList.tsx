'use client'

import { useMemo } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { buildRowModel } from '@/lib/results'
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
  pubDate?: string
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

  const models = useMemo<Array<{ model: RowModel; index: number }>>(() => {
    return results.rows.map((row, index) => ({
      model: buildRowModel(row, results.columns, results.source) as RowModel,
      index,
    }))
  }, [results])

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
