'use client'

import { useI18n } from '@/lib/app-i18n'
import { buildDocPreview } from '@/lib/docPreview'

export default function DocTab({ row }: { row: any }) {
  const { t } = useI18n()
  const preview = buildDocPreview(row.url)

  return (
    <div className="results-detail-card">
      <h3>{row.title}</h3>
      {row.meta.map((item: { label: string; value: string }) => (
        <div key={item.label} className="results-doc-meta">
          <span className="results-field-label">{item.label}</span>
          <span>{item.value}</span>
        </div>
      ))}
      {preview.mode === 'iframe' && (
        <iframe
          className="results-doc-frame"
          src={preview.src}
          title={row.title || 'document preview'}
        />
      )}
      {preview.mode === 'fallback' && (
        <div className="results-error">
          <p>{t('results.docNoPdf')}</p>
          <a href={preview.url} download rel="noopener noreferrer">
            {t('results.docPdfFallbackDownload')}
          </a>
        </div>
      )}
      {preview.mode === 'unavailable' && (
        <div className="results-error">
          <p>{t('results.docUnavailable')}</p>
        </div>
      )}
    </div>
  )
}
