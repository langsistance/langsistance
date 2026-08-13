'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useI18n } from '@/lib/app-i18n'
import { useChatSession, type ChatMessage } from '@/contexts/ChatContext'
import { useChatStream } from '@/lib/useChatStream'
import { getSession } from '@/services/api'
import ResultCard from '@/components/app/ResultCard'
import ResultList from '@/components/app/results/ResultList'
import DetailPanel from '@/components/app/results/DetailPanel'
import { buildRowModel } from '@/lib/results'

export default function ResultsPage() {
  const { t } = useI18n()
  const [activeRowId, setActiveRowId] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<string>('details')
  const router = useRouter()
  const searchParams = useSearchParams()
  const { messages, setMessages, sessionId, setSessionId, resultsSetId, setResultsSetId, input, setInput, streaming } = useChatSession()
  const { send, selectedFiles, setSelectedFiles, addFiles, removeFile, isDragOver, setIsDragOver } = useChatStream()

  const setId = searchParams.get('set') || resultsSetId
  const loadedRef = useRef(false)

  // Hydrate conversation when arriving with a session_id but no messages
  useEffect(() => {
    const sid = searchParams.get('session_id')
    if (!sid || loadedRef.current || messages.length > 0) return
    ;(async () => {
      try {
        const data = await getSession(sid)
        if (!Array.isArray(data.messages)) return
        // Only mark hydrated after the fetch succeeds so a transient failure
        // can retry on the next navigation/re-mount.
        loadedRef.current = true
        setMessages(data.messages
          .filter((m: any) => m.role && m.content)
          .map((m: any, i: number) => ({
            id: `hist_${i}_${Date.now()}`,
            role: m.role,
            content: m.content,
            taskId: m.taskId || undefined,
            resultSummary: m.resultSummary || undefined,
            patent_ids: m.patent_ids || undefined,
            results: m.results || undefined,
            artifacts: [],
          })))
        setSessionId(sid)
      } catch {
        // Session unavailable — stay in empty state
      }
    })()
  }, [searchParams, messages.length, setMessages, setSessionId])

  useEffect(() => {
    if (setId) setResultsSetId(setId)
  }, [setId, setResultsSetId])

  const activeMessage: ChatMessage | undefined = useMemo(() => {
    if (!setId) return undefined
    return messages.find((m) => (m as any).results?.setId === setId)
  }, [messages, setId])

  const activeRow = useMemo(() => {
    if (!activeMessage || !activeRowId) return null
    const results = (activeMessage as any).results
    if (!results) return null
    const row = results.rows.find((r: any) => buildRowModel(r, results.columns, results.source).id === activeRowId)
    return row ? buildRowModel(row, results.columns, results.source) : null
  }, [activeMessage, activeRowId])

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  if (!activeMessage) {
    return (
      <div className="page active results-page">
        <div className="results-empty">
          <h3>{t('results.emptyTitle')}</h3>
          <p>{t('results.emptyHint')}</p>
          <button onClick={() => router.push('/app/chat')}>{t('results.backToChat')}</button>
        </div>
      </div>
    )
  }

  return (
    <div className="page active results-page">
      <div className={`results-layout${activeRow ? ' with-detail' : ''}`}>
        <aside className="results-chat-sidebar">
          <div className="results-chat-messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`results-chat-item ${msg.role}`}>
                {msg.role === 'assistant' && (msg as any).results ? (
                  <ResultCard results={(msg as any).results} sessionId={sessionId} />
                ) : (
                  <span>{msg.content.length > 120 ? msg.content.slice(0, 120) + '…' : msg.content}</span>
                )}
              </div>
            ))}
          </div>
          <div className="results-chat-input">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t('results.sidebarPlaceholder')}
              rows={2}
            />
            <button onClick={() => send()} disabled={streaming || !input.trim()}>→</button>
          </div>
        </aside>
        <ResultList
          results={(activeMessage as any).results}
          activeRowId={activeRowId}
          onSelect={(model) => { setActiveRowId(model.id); setActiveTab(model.isDocument ? 'doc' : 'details') }}
          onOpenTab={(model, tab) => { setActiveRowId(model.id); setActiveTab(tab) }}
        />
        {activeRow && (
          <DetailPanel
            row={activeRow}
            tab={activeTab}
            onTabChange={setActiveTab}
            onClose={() => setActiveRowId(null)}
          />
        )}
      </div>
    </div>
  )
}
