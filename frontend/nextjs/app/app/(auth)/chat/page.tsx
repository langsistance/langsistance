'use client'

import { useRef, useEffect } from 'react'
import { usePathname, useSearchParams } from 'next/navigation'
import { getSession, saveSessionMessages, pollLongTaskBatchStatus, getLongTaskReportUrl } from '@/services/api'
import { replaceAssistantMessage, shouldResetConversationOnNavigation } from '@/lib/chatSession'
import { useI18n } from '@/lib/app-i18n'
import { useAuth } from '@/contexts/AuthContext'
import MarkdownMessage from '@/components/app/MarkdownMessage'
import UserCopyButton from '@/components/app/UserCopyButton'
import SceneHint from '@/components/app/SceneHint'
import { pruneResultsForPersistence } from '@/lib/results'
import { loadResultsStore, restoreResultsInMessages } from '@/lib/resultsStore'
import { useChatSession } from '@/contexts/ChatContext'
import { useChatStream } from '@/lib/useChatStream'

export default function Chat() {
  const { t, lang } = useI18n()
  const { user, requireAuth } = useAuth()
  const {
    messages,
    setMessages,
    input,
    setInput,
    streaming,
    streamingId,
    sessionId,
    setSessionId,
  } = useChatSession()
  const {
    send,
    abort,
    transientStatus,
    selectedFiles,
    setSelectedFiles,
    addFiles,
    removeFile,
    isDragOver,
    setIsDragOver,
    stopLongTaskPolling,
    startLongTaskPolling,
  } = useChatStream()
  const searchParams = useSearchParams()
  const sessionLoadedRef = useRef(false)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const chatContainerRef = useRef<HTMLDivElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const isNearBottomRef = useRef(true)
  const pendingQuerySentRef = useRef<string | null>(null)

  // Reset the auto-growing textarea height after a send empties the input.
  // The height reset previously lived inside the chat send() pipeline, which
  // has moved into the useChatStream() hook (which cannot reach this DOM ref).
  useEffect(() => {
    if (!input && textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [input])

  // Auto-send a query arriving via URL (?pending_query=...) — used by the
  // results page's 审查历史 button to trigger prosecution analysis back in
  // the conversation.  The parameter is cleared from the URL so a refresh
  // never re-sends it.
  useEffect(() => {
    const pending = searchParams.get('pending_query')
    if (!pending || pendingQuerySentRef.current === pending) return
    pendingQuerySentRef.current = pending
    const url = new URL(window.location.href)
    url.searchParams.delete('pending_query')
    window.history.replaceState({}, '', url.toString())
    send([], pending)
  }, [searchParams, send])



  // Load session from URL param (and resume long task polling if needed)
  const lastLoadedSidRef = useRef<string | null>(null)
  const pathname = usePathname()
  useEffect(() => {
    const sid = searchParams.get('session_id')
    if (!sid) {
      // A URL without session_id means "new conversation" — but only when
      // the chat page is the active route.  Handles both the case where
      // the component persisted across a client-side navigation (e.g.
      // 新对话) and the case where it freshly mounted with stale
      // ChatProvider state from the parent layout.  During a client-side
      // transition away (e.g. the auto-open of the results page after a
      // search) the router context updates before this component
      // unmounts — clearing then would wipe the shared conversation out
      // from under the incoming page.
      if (!shouldResetConversationOnNavigation(pathname)) return
      // [DIAG]
      try {
        const key = 'copiioai_diag'
        sessionStorage.setItem(key, (sessionStorage.getItem(key) || '') + '|C:clear')
        console.log('[copiioai-diag] chat-clear fired breadcrumb=', sessionStorage.getItem(key))
      } catch {}
      stopLongTaskPolling()
      setMessages([])
      setSessionId(null)
      lastLoadedSidRef.current = null
      sessionLoadedRef.current = false
      return
    }
    if (sid === lastLoadedSidRef.current) return

    lastLoadedSidRef.current = sid
    sessionLoadedRef.current = true

    let cancelled = false
    ;(async () => {
      try {
        const data = await getSession(sid)
        if (cancelled) return
        const longTaskIds: string[] = data.long_task_ids || []
        if (data.messages && Array.isArray(data.messages)) {
          const loaded = data.messages
            .filter((m: { role: string; content: string }) => m.role && m.content)
            .map((m: { role: string; content: string; taskId?: string; patent_ids?: string[] }, i: number) => ({
              id: `hist_${i}_${Date.now()}`,
              role: m.role,
              content: m.content,
              taskId: (m as any).taskId || undefined,
              artifacts: [],
              resultSummary: (m as any).resultSummary || undefined,
              patent_ids: (m as any).patent_ids || undefined,
              results: (m as any).results || undefined,
            }))
            // Strip orphan long-task messages (🔬/✅/❌ without taskId).
            // These were saved before taskId was attached during SSE.
            // The resume loop below will recreate them with proper taskId,
            // avoiding duplicates that never update.
            .filter((m: { taskId?: string; content: string }) =>
              m.taskId || (!m.content.includes('🔬') && !m.content.includes('✅') && !m.content.includes('❌'))
            )
          if (loaded.length > 0) {
            // [DIAG]
            try {
              const key = 'copiioai_diag'
              sessionStorage.setItem(key, (sessionStorage.getItem(key) || '') + `|C:load=${loaded.length}`)
              console.log('[copiioai-diag] chat-session-load replacing with', loaded.length, 'breadcrumb=', sessionStorage.getItem(key))
            } catch {}
            setMessages(restoreResultsInMessages(loaded, loadResultsStore(window.localStorage)))
            // Scroll to bottom after loading session messages
            requestAnimationFrame(() => {
              isNearBottomRef.current = true
              bottomRef.current?.scrollIntoView({ behavior: 'instant' as ScrollBehavior })
            })
          }
        }
        setSessionId(sid)

        // Resume polling for any incomplete long tasks — batch fetch all statuses
        if (longTaskIds.length > 0) {
          try {
            const batch = await pollLongTaskBatchStatus(longTaskIds)
            for (const tid of longTaskIds) {
              const status = batch[tid]
              if (!status) continue

            // Session save happens ~1s after SSE end, but the task may complete
            // minutes later.  The in-memory message transitions to ✅/❌ via
            // polling, but the saved session still has the stale 🔬 content.
            // Update completed/failed messages so the card shows the final state
            // and so send()'s filter (which checks for ✅/❌) preserves them.
            if (status && (status.status === 'completed' || status.status === 'success')) {
              const files = (status.report_files || [])
                .map((f: { format: string }) =>
                  `[${f.format.toUpperCase()}](${getLongTaskReportUrl(tid, f.format as 'pdf' | 'docx')})`)
                .join(' | ')
              setMessages(m => {
                const idx = m.findIndex(msg => msg.taskId === tid)
                const nextContent = t('chat.longTaskCompleted').replace('{files}', files)
                if (idx >= 0) {
                  return m.map((msg, i) => i === idx
                    ? { ...msg, content: nextContent, resultSummary: status.result_summary || msg.resultSummary }
                    : msg)
                }
                return [...m, {
                  id: `lt_resume_${tid}`, role: 'assistant',
                  content: nextContent,
                  artifacts: [], taskId: tid,
                  resultSummary: status.result_summary,
                }]
              })
              continue
            }
            if (status && (status.status === 'failed' || status.status === 'error')) {
              setMessages(m => {
                const idx = m.findIndex(msg => msg.taskId === tid)
                if (idx >= 0) {
                  return replaceAssistantMessage(m, m[idx].id,
                    `${t('chat.longTaskFailed')} ${status.error_message || ''}`)
                }
                return [...m, {
                  id: `lt_resume_${tid}`, role: 'assistant',
                  content: `${t('chat.longTaskFailed')} ${status.error_message || ''}`,
                  artifacts: [], taskId: tid,
                }]
              })
              continue
            }

            if (!status || status.status === 'unknown') {
              continue
            }

            const phaseLabel = status.current_step || status.current_phase || ''
            const progress = status.progress != null ? `[${status.progress}%]` : ''
            const progressContent = ((progress || phaseLabel)
              ? t('chat.longTaskProgress')
                  .replace('{progress}', progress)
                  .replace('{phase}', phaseLabel)
              : t('chat.longTaskRunning')) + ` Task ID: ${tid}`
            let pollMsgId = `lt_resume_${tid}`
            setMessages(m => {
              const existingIdx = m.findIndex(msg => msg.taskId === tid)
              if (existingIdx >= 0) {
                pollMsgId = m[existingIdx].id
                return m.map((msg, i) => i === existingIdx
                  ? {
                      ...msg,
                      content: progressContent,
                      resultSummary: status.result_summary || msg.resultSummary,
                    }
                  : msg)
              }
              return [...m, {
                id: pollMsgId,
                role: 'assistant',
                content: progressContent,
                artifacts: [],
                taskId: tid,
                resultSummary: status.result_summary,
              }]
            })
            startLongTaskPolling(tid, pollMsgId)
            }
          } catch {
            // Batch status fetch failed — skip resume
          }
        }
      } catch {
        // Session not found or error — start fresh
      }
    })()

    return () => { cancelled = true }
  }, [searchParams, pathname, sessionId, setMessages, setSessionId])

  const messagesHash = JSON.stringify(messages.map(m => ({ role: m.role, content: m.content, taskId: (m as any).taskId, resultSummary: (m as any).resultSummary, patent_ids: (m as any).patent_ids })))
  // Save session after streaming completes — but ONLY if a session already exists
  // (session is created only when a long task is triggered)
  const pendingSaveRef = useRef(false)
  useEffect(() => {
    if (streaming || messages.length === 0) return
    if (!sessionId) return  // No session yet = no long task ever triggered
    if (pendingSaveRef.current) return
    pendingSaveRef.current = true

    const timer = setTimeout(async () => {
      try {
        const toSave = messages.map(m => ({
          role: m.role,
          content: m.content,
          ...(m.taskId ? { taskId: m.taskId } : {}),
          ...(m.resultSummary ? { resultSummary: m.resultSummary } : {}),
          ...(m.patent_ids ? { patent_ids: m.patent_ids } : {}),
          ...((m as any).results
            ? { results: pruneResultsForPersistence((m as any).results) }
            : {}),
        }))
        await saveSessionMessages(sessionId, toSave)
      } catch {
        // Non-critical
      }
      pendingSaveRef.current = false
    }, 1000)

    return () => { clearTimeout(timer); pendingSaveRef.current = false }
  }, [streaming, messagesHash, sessionId])

  // Track whether the user is scrolled near the bottom of the chat.
  useEffect(() => {
    const container = chatContainerRef.current
    if (!container) return
    function handleScroll() {
      if (!container) return
      const threshold = 80 // px from bottom considered "near bottom"
      isNearBottomRef.current =
        container.scrollHeight - container.scrollTop - container.clientHeight <= threshold
    }
    container.addEventListener('scroll', handleScroll, { passive: true })
    return () => container.removeEventListener('scroll', handleScroll)
  }, [])

  // Auto-scroll to bottom when messages change, but only if the user is
  // already near the bottom. Use instant scroll during streaming to avoid
  // overlapping smooth animations that cause jitter.
  useEffect(() => {
    if (!isNearBottomRef.current) return
    bottomRef.current?.scrollIntoView({ behavior: streaming ? 'instant' : 'smooth' })
  }, [messages, streaming])

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
  }

  function handleFilePaste(e: React.ClipboardEvent) {
    const items = e.clipboardData?.files
    if (items && items.length > 0) {
      e.preventDefault()
      addFiles(items)
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files)
    }
  }

  function openFilePicker() {
    fileInputRef.current?.click()
  }

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

  return (
    <div className="page active">
      <div className="chat-container">
        <div className="chat-messages" ref={chatContainerRef}>
          {messages.length === 0 && (
            <div className="chat-message-wrapper">
              <div className="empty-state">
                <h3>{t('chat.welcome.greeting')}</h3>
                <p>{t('chat.welcome.prompt')}</p>
              </div>
            </div>
          )}
          <SceneHint />
          {messages.map((msg) => (
            <div key={msg.id} className={`chat-message-wrapper ${msg.role}`}>
              {msg.role === 'assistant' ? (
                <>
                  <MarkdownMessage
                    content={msg.content}
                    artifacts={msg.artifacts || []}
                    resultSummary={msg.resultSummary}
                    streaming={streaming && streamingId === msg.id}
                    transientStatus={streaming && streamingId === msg.id ? transientStatus : ''}
                    analysisType={(msg as any).analysisType}
                    tableColumns={(msg as any).tableColumns}
                    familyOverview={(msg as any).familyOverview}
                    jurisdictions={(msg as any).jurisdictions}
                  />
                </>
              ) : (
                <div className="chat-message user">
                  {msg.content}
                  <UserCopyButton content={msg.content} />
                  <div className="user-copy-button-bridge" />
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {isDragOver && (
          <div
            className="file-drop-overlay"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="file-drop-zone">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
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
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              className="file-input-hidden"
              accept=".pdf,.docx,.xml,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/xml,text/xml"
              multiple
              onChange={e => { if (e.target.files) addFiles(e.target.files); e.target.value = '' }}
            />
            <button
              className="file-upload-btn"
              onClick={openFilePicker}
              aria-label="Attach patent files"
              title={t('chat.attachFiles') || 'Attach patent specification files (PDF, DOCX, XML)'}
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
              </svg>
            </button>
            <textarea
              ref={textareaRef}
              className="chat-input"
              value={input}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              onPaste={handleFilePaste}
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
      </div>
    </div>
  )
}
