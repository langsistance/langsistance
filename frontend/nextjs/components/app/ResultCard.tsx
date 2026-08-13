'use client'

import { useRouter } from 'next/navigation'
import { useI18n } from '@/lib/app-i18n'

interface ResultsPayload {
  setId: string
  source: string
  columns: Array<{ key: string; label: string; role: string }>
  rows: Array<Record<string, unknown>>
}

export default function ResultCard({ results, sessionId }: { results: ResultsPayload; sessionId: string | null }) {
  const { t } = useI18n()
  const router = useRouter()

  function openResultsPage() {
    const params = new URLSearchParams({ set: results.setId })
    if (sessionId) params.set('session_id', sessionId)
    router.push(`/app/results?${params.toString()}`)
  }

  return (
    <div className="result-card">
      <div className="result-card-header">
        <span className="result-card-title">
          {t('chat.resultsCardTitle').replace('{count}', String(results.rows.length))}
        </span>
        <span className="result-card-source">{results.source}</span>
      </div>
      <button className="result-card-button" onClick={openResultsPage}>
        {t('chat.resultsViewButton')}
      </button>
    </div>
  )
}
