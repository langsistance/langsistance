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
    // Publication number first for spec: the Baiten PDF proxy needs
    // pub_num + pub_date (both ride on CN candidates), USPTO rows carry
    // patentNumber/applicationNumberText natively.
    const identifier = row.patentId || row.applicationNumber
    if (!identifier) {
      setState('error')
      return
    }
    fetchPatentSpec(row.source, identifier, row.pubDate)
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
  if (state === 'error' || !data?.pdf_url) {
    return (
      <div className="results-detail-card results-error">
        <p>{t('results.specError')}</p>
        <button onClick={() => setRetryKey((k) => k + 1)}>{t('results.retry')}</button>
      </div>
    )
  }

  // The backend returns the /uspto/download proxy URL (PDF).  inline=1
  // makes the proxy serve the file inline (and coerce octet-stream to
  // application/pdf for the embedded viewer); the navpanes fragment keeps
  // the thumbnail pane collapsed — same presentation as document rows.
  const src = `${data.pdf_url}${data.pdf_url.includes('?') ? '&' : '?'}inline=1#navpanes=0`

  return (
    <div className="results-detail-card">
      <iframe
        className="results-doc-frame"
        src={src}
        title={t('results.specTab')}
      />
    </div>
  )
}
