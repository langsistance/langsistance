'use client'

import { useEffect, useState } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { fetchPatentClaims, type PatentClaimsResponse } from '@/services/api'
import { copyTextToClipboard } from '@/lib/clipboard'

export default function ClaimsTab({ row }: { row: any }) {
  const { t } = useI18n()
  const [state, setState] = useState<'loading' | 'error' | 'data'>('loading')
  const [data, setData] = useState<PatentClaimsResponse | null>(null)
  const [retryKey, setRetryKey] = useState(0)
  const [copiedNumber, setCopiedNumber] = useState<number | null>(null)

  useEffect(() => {
    let cancelled = false
    setState('loading')
    const identifier = row.patentId || row.applicationNumber
    if (!identifier) {
      setState('error')
      return
    }
    fetchPatentClaims(row.source, identifier)
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

  if (state === 'loading') return <div className="results-detail-card">{t('results.claimsLoading')}</div>
  if (state === 'error' || !data) {
    return (
      <div className="results-detail-card results-error">
        <p>{t('results.claimsError')}</p>
        <button onClick={() => setRetryKey((k) => k + 1)}>{t('results.retry')}</button>
      </div>
    )
  }

  async function handleCopy(number: number, text: string) {
    if (await copyTextToClipboard(text)) {
      setCopiedNumber(number)
      setTimeout(() => setCopiedNumber(null), 2000)
    }
  }

  return (
    <div className="results-detail-card">
      {data.claims.map((claim) => (
        <div key={claim.number} className={`results-claim${claim.independent ? ' independent' : ''}`}>
          <div className="results-claim-header">
            <span>{claim.independent ? t('results.claimIndependent') : `#${claim.number}`}</span>
            <button onClick={() => handleCopy(claim.number, claim.text)}>
              {copiedNumber === claim.number ? '✓' : t('results.claimCopy')}
            </button>
          </div>
          <p>{claim.text}</p>
        </div>
      ))}
    </div>
  )
}
