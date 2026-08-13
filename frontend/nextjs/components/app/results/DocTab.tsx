'use client'

import { useI18n } from '@/lib/app-i18n'

export default function DocTab({ row }: { row: any }) {
  const { t } = useI18n()
  return (
    <div className="results-detail-card">
      <h3>{row.title}</h3>
      {row.meta.map((item: { label: string; value: string }) => (
        <div key={item.label} className="results-doc-meta">
          <span className="results-field-label">{item.label}</span>
          <span>{item.value}</span>
        </div>
      ))}
      {row.url && (
        <a className="results-doc-link" href={row.url} target="_blank" rel="noopener noreferrer">
          {t('results.rowDocView')}
        </a>
      )}
    </div>
  )
}
