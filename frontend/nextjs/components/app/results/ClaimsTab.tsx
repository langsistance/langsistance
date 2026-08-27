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
    // Application number first: CN rows carry the Baiten app_num (the
    // backend claims branch uses it directly — the getDoc resolution hop
    // is broken), and USPTO rows carry applicationNumberText natively.
    const identifier = row.applicationNumber || row.patentId
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

  // No structured claims (image-only scanned document) — the backend
  // returns the /uspto/download proxy URL; show the original PDF in the
  // inline viewer, same presentation as the spec tab.
  if (data.pdf_url) {
    const src = `${data.pdf_url}${data.pdf_url.includes('?') ? '&' : '?'}inline=1#navpanes=0`
    return (
      <div className="results-detail-card">
        <iframe
          className="results-doc-frame"
          src={src}
          title={t('results.claimsTab')}
        />
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
      {(data.claims || []).map((claim) => (
        <div
          key={claim.number}
          className={`results-claim${claim.independent ? ' independent' : ''}${claim.status === 'canceled' ? ' canceled' : ''}`}
        >
          <div className="results-claim-header">
            <span>
              {`#${claim.number}`}
              {claim.status === 'canceled' && ` · ${t('results.claimCanceled')}`}
              {claim.status !== 'canceled' && claim.independent && ` · ${t('results.claimIndependent')}`}
            </span>
            {claim.text && (
              <button onClick={() => handleCopy(claim.number, claim.text)}>
                {copiedNumber === claim.number ? '✓' : t('results.claimCopy')}
              </button>
            )}
          </div>
          {claim.text ? <p>{claim.text}</p> : <p className="results-claim-canceled">({t('results.claimCanceled')})</p>}
        </div>
      ))}
    </div>
  )
}
