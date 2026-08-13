'use client'

import { useI18n } from '@/lib/app-i18n'

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

export default function ResultRow({
  model, active, onSelect, onOpenTab, onProsecution,
}: {
  model: RowModel
  active: boolean
  onSelect: (model: RowModel) => void
  onOpenTab: (model: RowModel, tab: string) => void
  onProsecution: (model: RowModel) => void
}) {
  const { t } = useI18n()

  return (
    <div className={`result-row${active ? ' active' : ''}`} onClick={() => onSelect(model)}>
      <button
        className="result-row-title"
        onClick={(e) => {
          e.stopPropagation()
          onSelect(model)
        }}
      >
        {model.title || '—'}
      </button>
      <div className="result-row-meta">
        {model.meta.map((item) => item.value).filter(Boolean).join(' · ')}
      </div>
      <div className="result-row-actions" onClick={(e) => e.stopPropagation()}>
        {model.isDocument ? (
          <>
            {model.url && (
              <>
                <a href={model.url} target="_blank" rel="noopener noreferrer">
                  {t('results.rowDocView')}
                </a>
                <a href={model.url} download rel="noopener noreferrer">
                  {t('results.rowDocDownload')}
                </a>
              </>
            )}
          </>
        ) : (
          <>
            <button onClick={() => onOpenTab(model, 'spec')}>{t('results.rowSpec')}</button>
            <button onClick={() => onOpenTab(model, 'claims')}>{t('results.rowClaims')}</button>
            <button onClick={() => onProsecution(model)}>{t('results.rowProsecution')}</button>
          </>
        )}
      </div>
    </div>
  )
}
