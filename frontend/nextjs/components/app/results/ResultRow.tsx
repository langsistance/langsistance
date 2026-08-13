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
  model, active, onSelect, onOpenTab,
}: {
  model: RowModel
  active: boolean
  onSelect: (model: RowModel) => void
  onOpenTab: (model: RowModel, tab: string) => void
}) {
  const { t } = useI18n()

  return (
    <div className={`result-row${active ? ' active' : ''}`} onClick={() => onSelect(model)}>
      <div className="result-row-title">{model.title || '—'}</div>
      <div className="result-row-meta">
        {model.meta.map((item) => (
          <span key={item.label} className="result-row-meta-item">
            {item.label}: {item.value}
          </span>
        ))}
      </div>
      <div className="result-row-actions" onClick={(e) => e.stopPropagation()}>
        {model.isDocument ? (
          <>
            <button onClick={() => onOpenTab(model, 'doc')}>{t('results.rowDocInfo')}</button>
            {model.url && (
              <a href={model.url} target="_blank" rel="noopener noreferrer">
                {t('results.rowDocView')}
              </a>
            )}
          </>
        ) : (
          <>
            <button onClick={() => onOpenTab(model, 'details')}>{t('results.rowDetails')}</button>
            <button onClick={() => onOpenTab(model, 'spec')}>{t('results.rowSpec')}</button>
            <button onClick={() => onOpenTab(model, 'claims')}>{t('results.rowClaims')}</button>
            <button onClick={() => onOpenTab(model, 'prosecution')}>{t('results.rowProsecution')}</button>
          </>
        )}
      </div>
    </div>
  )
}
