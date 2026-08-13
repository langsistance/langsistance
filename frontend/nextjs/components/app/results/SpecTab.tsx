'use client'

import { useEffect, useState } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { fetchPatentSpec, type PatentSpecResponse } from '@/services/api'

export default function SpecTab({ row }: { row: any }) {
  const { t } = useI18n()
  const [state, setState] = useState<'loading' | 'error' | 'data'>('loading')
  const [data, setData] = useState<PatentSpecResponse | null>(null)
  const [retryKey, setRetryKey] = useState(0)

  useEffect(() => {
    let cancelled = false
    setState('loading')
    const identifier = row.patentId || row.applicationNumber
    if (!identifier) {
      setState('error')
      return
    }
    fetchPatentSpec(row.source, identifier)
      .then((payload) => {
        if (cancelled) return
        setData(payload)
        setState('data')
      })
      .catch(() => {
        if (!cancelled) setState('error')
      })
    return () => { cancelled = true }
  }, [row.source, row.patentId, row.applicationNumber, retryKey])

  if (state === 'loading') return <div className="results-detail-card">{t('results.specLoading')}</div>
  if (state === 'error' || !data) {
    return (
      <div className="results-detail-card results-error">
        <p>{t('results.specError')}</p>
        <button onClick={() => setRetryKey((k) => k + 1)}>{t('results.retry')}</button>
      </div>
    )
  }

  return (
    <div className="results-detail-card">
      {data.source_url && (
        <a className="results-spec-pdf" href={data.source_url} target="_blank" rel="noopener noreferrer">
          {t('results.specPdfLink')}
        </a>
      )}
      {data.sections.map((section) => (
        <section key={section.heading} className="results-spec-section">
          <h4>{section.heading}</h4>
          {section.paragraphs.map((para, index) => (
            <p key={index}>{para}</p>
          ))}
        </section>
      ))}
    </div>
  )
}
