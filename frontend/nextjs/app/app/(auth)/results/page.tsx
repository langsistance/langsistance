'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { useI18n } from '@/lib/app-i18n'
import { useChatSession, type ChatMessage } from '@/contexts/ChatContext'
import { useChatStream } from '@/lib/useChatStream'
import { getSession } from '@/services/api'
import MarkdownMessage from '@/components/app/MarkdownMessage'
import UserCopyButton from '@/components/app/UserCopyButton'
import SceneHint from '@/components/app/SceneHint'
import ResultList from '@/components/app/results/ResultList'
import DetailPanel from '@/components/app/results/DetailPanel'
import { buildRowModel, resolveActiveResultsMessage } from '@/lib/results'

function getFileTypeBadge(file: File): string {
  const ext = '.' + file.name.split('.').pop()?.toLowerCase()
  if (ext === '.docx') return 'DOCX'
  if (ext === '.xml') return 'XML'
  return 'PDF'
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

export default function ResultsPage() {
  const { t } = useI18n()
  const [activeRowId, setActiveRowId] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<string>('details')
  const [listCollapsed, setListCollapsed] = useState(false)
  const router = useRouter()
  const searchParams = useSearchParams()
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const { messages, setMessages, sessionId, setSessionId, resultsSetId, setResultsSetId, input, setInput, streaming, streamingId } = useChatSession()
  const { send, abort, transientStatus, selectedFiles, setSelectedFiles, addFiles, removeFile, isDragOver, setIsDragOver } = useChatStream()

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

  // New result set → the list always starts visible
  useEffect(() => {
    setListCollapsed(false)
  }, [setId])

  const activeMessage: ChatMessage | undefined = useMemo(() => {
    // Fall back to the newest results message when the URL's set has no
    // exact match — the auto-navigation state race right after streaming
    // can briefly render the results page before the newest message's
    // results commit (intermittent empty page).
    return resolveActiveResultsMessage(messages, setId) || undefined
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

  function handleProsecution(model: any) {
    // 审查历史 analysis runs in the conversation page — navigate back with
    // a pending query so the normal chat pipeline (intent routing → long
    // task card) picks it up exactly as if the user had typed it.
    const identifier = model.patentId || model.applicationNumber
    const isUsAppNumber = /^\d{8}$/.test(model.applicationNumber || '')
    const queryText = isUsAppNumber
      ? `分析专利 ${identifier} 的审查历史`
      : `分析 ${identifier} 及其全球同族申请的审查差异`
    const params = new URLSearchParams()
    if (sessionId) params.set('session_id', sessionId)
    params.set('pending_query', queryText)
    router.push(`/app/chat?${params.toString()}`)
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
      <div className={`results-layout${listCollapsed ? ' collapsed' : ''}`}>
        <aside className="results-chat-sidebar">
          <div className="chat-messages">
            {messages.length === 0 && (
              <div className="empty-state">
                <h3>{t('chat.welcome.greeting')}</h3>
                <p>{t('chat.welcome.prompt')}</p>
              </div>
            )}
            <SceneHint />

            {messages.map((msg) => (
              <div key={msg.id} className={`chat-message-wrapper ${msg.role}`}>
                {msg.role === 'assistant' ? (
                  <MarkdownMessage
                    content={msg.content}
                    artifacts={msg.artifacts || []}
                    resultSummary={msg.resultSummary}
                    streaming={streaming && streamingId === msg.id}
                    transientStatus={streaming && streamingId === msg.id ? transientStatus : ''}
                  />
                ) : (
                  <div className="chat-message user">
                    {msg.content}
                    <UserCopyButton content={msg.content} />
                    <div className="user-copy-button-bridge" />
                  </div>
                )}
              </div>
            ))}
          </div>
          {isDragOver && (
            <div
              className="file-drop-overlay"
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
              onDragLeave={(e) => { e.preventDefault(); setIsDragOver(false) }}
              onDrop={(e) => { e.preventDefault(); setIsDragOver(false); if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files) }}
            >
              <div className="file-drop-zone">
                <p>{t('chat.dropFilesHere') || 'Drop patent specification files here'}</p>
                <span className="file-drop-hint">PDF, DOCX, XML · Max 10 MB each · Up to 100 files</span>
              </div>
            </div>
          )}
          <div className="chat-input-container">
            {selectedFiles.length > 0 && (
              <div className="file-chips-bar">
                {selectedFiles.map((file, i) => (
                  <div key={`${file.name}-${i}`} className="file-chip">
                    <span className={`file-chip-badge ${getFileTypeBadge(file).toLowerCase()}`}>
                      {getFileTypeBadge(file)}
                    </span>
                    <span className="file-chip-name">{file.name}</span>
                    <span className="file-chip-size">{formatFileSize(file.size)}</span>
                    <button
                      className="file-chip-remove"
                      onClick={() => removeFile(i)}
                      aria-label="Remove file"
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  </div>
                ))}
              </div>
            )}
            <div
              className="chat-input-wrapper"
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
              onDragLeave={(e) => { e.preventDefault(); setIsDragOver(false) }}
              onDrop={(e) => { e.preventDefault(); setIsDragOver(false); if (e.dataTransfer.files.length > 0) addFiles(e.dataTransfer.files) }}
            >
              <input
                ref={fileInputRef}
                type="file"
                className="file-input-hidden"
                accept=".pdf,.docx,.xml"
                multiple
                onChange={(e) => { if (e.target.files) addFiles(e.target.files); e.target.value = '' }}
              />
              <button
                className="file-upload-btn"
                onClick={() => fileInputRef.current?.click()}
                aria-label="Attach patent files"
                title={t('chat.attachFiles') || 'Attach patent specification files (PDF, DOCX, XML)'}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                </svg>
              </button>
              <textarea
                className="chat-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={t('chat.placeholder')}
                rows={1}
              />
              {streaming ? (
                <button
                  className="send-btn"
                  onClick={abort}
                  style={{ background: 'var(--color-text-secondary)' }}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <rect x="6" y="6" width="12" height="12" />
                  </svg>
                </button>
              ) : (
                <button
                  className="send-btn"
                  onClick={() => send()}
                  disabled={!input.trim() && selectedFiles.length === 0}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="22" y1="2" x2="11" y2="13" />
                    <polygon points="22 2 15 22 11 13 2 9 22 2" />
                  </svg>
                </button>
              )}
            </div>
          </div>
        </aside>
        {listCollapsed && (
          <button
            className="results-expand-btn"
            onClick={() => setListCollapsed(false)}
            title={t('results.expandList')}
          >
            ⟩ {t('results.expandList')}
          </button>
        )}
        {!listCollapsed && (
        <main className="results-main">
          <ResultList
            results={(activeMessage as any).results}
            activeRowId={activeRowId}
            onSelect={(model) => { setActiveRowId(model.id); setActiveTab(model.isDocument ? 'doc' : 'details') }}
            onOpenTab={(model, tab) => { setActiveRowId(model.id); setActiveTab(tab) }}
            onProsecution={handleProsecution}
            onCollapse={() => setListCollapsed(true)}
          />
          {activeRow && (
            <div className="results-overlay">
              <DetailPanel
                row={activeRow}
                tab={activeTab}
                onTabChange={setActiveTab}
                onClose={() => setActiveRowId(null)}
              />
            </div>
          )}
        </main>
        )}
      </div>
    </div>
  )
}
