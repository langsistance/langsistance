'use client'

import { useRef, useEffect } from 'react'
import { usePathname, useSearchParams } from 'next/navigation'
import { getSession, saveSessionMessages, pollLongTaskBatchStatus, getLongTaskReportUrl, retryLongTask, authHeaders } from '@/services/api'
import { replaceAssistantMessage, shouldResetConversationOnNavigation } from '@/lib/chatSession'
import { loadLastSession, clearLastSession } from '@/lib/chatStore'
import { useI18n } from '@/lib/app-i18n'
import { useAuth } from '@/contexts/AuthContext'
import MarkdownMessage from '@/components/app/MarkdownMessage'
import UserCopyButton from '@/components/app/UserCopyButton'
import ChatLanding from '@/components/app/ChatLanding'
import ChatComposer from '@/components/app/ChatComposer'
import { pruneResultsForPersistence } from '@/lib/results'
import { loadResultsStore, restoreResultsInMessages } from '@/lib/resultsStore'
import { useChatSession } from '@/contexts/ChatContext'
import { useChatStream } from '@/lib/useChatStream'

export default function Chat() {
  const { t, lang } = useI18n()
  const { user, requireAuth } = useAuth()
  const {
    messages,
    hydrated,
    setMessages,
    input,
    setInput,
    streaming,
    streamingId,
    sessionId,
    setSessionId,
    lastLoadedSidRef,
  } = useChatSession()
  const {
    send,
    abort,
    statusSteps,
    statusElapsed,
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
  const isNearBottomRef = useRef(true)
  const pendingQuerySentRef = useRef<string | null>(null)

  // Empty conversation shows the ChatLanding empty state (slogan + centered
  // composer + six capabilities).  send() adds user+assistant messages
  // synchronously, so any send flips straight back into normal chat mode.
  const showLanding = hydrated && messages.length === 0

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



  // 需求 2: restore a backend session into the chat.  Shared by the URL
  // branch and the last-session localStorage fallback so pure-chat
  // sessions restore exactly like long-task ones.  *cancelledRef* is set
  // by the effect cleanup when the user navigates away mid-restore.
  async function loadSessionIntoState(sid: string, cancelledRef: { current: boolean }) {
    console.info(`[sessdbg] LOAD-START sid=${sid}`)
    try {
      const data = await getSession(sid)
      if (cancelledRef.current) return
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
        const last = loaded[loaded.length - 1]
        console.info(`[sessdbg] LOAD-DONE sid=${sid} raw=${data.messages.length} msgs=${loaded.length} last=${last?.role} prefix=${JSON.stringify((last?.content || '').slice(0, 50))} longTasks=${longTaskIds.length}`)
        if (loaded.length > 0) {
          setMessages(restoreResultsInMessages(loaded, loadResultsStore(window.localStorage)))
          // Scroll to bottom after loading session messages
          requestAnimationFrame(() => {
            isNearBottomRef.current = true
            bottomRef.current?.scrollIntoView({ behavior: 'instant' as ScrollBehavior })
          })
        }
      }
      setSessionId(sid)
      // Sync the URL so a later refresh/back-forward keeps this session.
      // Next.js intercepts history.replaceState, so useSearchParams will
      // reflect the sid — the lastLoadedSidRef guard prevents the effect
      // from re-restoring (and clobbering) what we just loaded.
      const url = new URL(window.location.href)
      if (url.searchParams.get('session_id') !== sid) {
        url.searchParams.set('session_id', sid)
        window.history.replaceState({}, '', url.toString())
      }

      // Resume polling for any incomplete long tasks — batch fetch all statuses
      if (longTaskIds.length > 0) {
        try {
          const batch = await pollLongTaskBatchStatus(longTaskIds)
          if (cancelledRef.current) return
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
    } catch (e) {
      console.warn(`[sessdbg] LOAD-FAIL sid=${sid}`, e)
      // Session not found or error — start fresh
    }
  }

  // Load session from URL param (and resume long task polling if needed)
  const pathname = usePathname()
  useEffect(() => {
    const sid = searchParams.get('session_id')
    // [sessdbg] Full URL at mount — distinguishes a refresh on /app/chat
    // (expect MOUNT-SID) from a refresh on /app/results (no mount log here).
    console.info(`[sessdbg] MOUNT-EFFECT href=${window.location.href.split('?')[0]}?${window.location.search} sid=${sid ?? 'null'} lastLoadedSidRef=${lastLoadedSidRef.current ?? 'null'} msgs=${messages.length}`)
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
      stopLongTaskPolling()
      setMessages([])
      setSessionId(null)
      lastLoadedSidRef.current = null
      sessionLoadedRef.current = false
      // 需求 2: a fresh mount with no URL session restores the most recent
      // conversation from the backend.  Skipped when a conversation is
      // already in state (新对话 navigation) or the stored session belongs
      // to a different account.
      if (messages.length === 0 && user?.uid) {
        const last = loadLastSession(window.localStorage)
        console.info(`[sessdbg] MOUNT-NOSID last=${last ? `sid=${last.sid} uid=${last.uid}` : 'none'} uid=${user.uid}`)
        if (last && last.uid === user.uid) {
          const cancelledRef = { current: false }
          lastLoadedSidRef.current = last.sid
          sessionLoadedRef.current = true
          loadSessionIntoState(last.sid, cancelledRef)
          return () => { cancelledRef.current = true }
        }
      }
      return
    }
    if (sid === lastLoadedSidRef.current) return

    console.info(`[sessdbg] MOUNT-SID sid=${sid} lastLoadedSidRef=${lastLoadedSidRef.current ?? 'null'}`)
    lastLoadedSidRef.current = sid
    sessionLoadedRef.current = true

    const cancelledRef = { current: false }
    loadSessionIntoState(sid, cancelledRef)

    return () => { cancelledRef.current = true }
  }, [searchParams, pathname, sessionId, user, setMessages, setSessionId])

  // 需求 4: 失败卡片一键重试 — 重新提交任务, 并把卡片切换为进行中状态
  const handleRetryTask = async (taskId: string): Promise<boolean> => {
    try {
      const res = await retryLongTask(taskId)
      if (!res.success || !res.task_id) return false
      const newTaskId = res.task_id
      const target = messages.find(m => m.taskId === taskId)
      const targetId = target?.id ?? `lt_retry_${newTaskId}`
      setMessages(m => m.map(msg =>
        msg.taskId === taskId
          ? {
              ...msg,
              taskId: newTaskId,
              content: t('chat.longTaskProgress')
                  .replace('{progress}', '[0%]')
                  .replace('{phase}', t('chat.phasePreparing'))
                + ` Task ID: ${newTaskId}`,
            }
          : msg
      ))
      startLongTaskPolling(newTaskId, targetId)
      return true
    } catch {
      return false
    }
  }

  const messagesHash = JSON.stringify(messages.map(m => ({ role: m.role, content: m.content, taskId: (m as any).taskId, resultSummary: (m as any).resultSummary, patent_ids: (m as any).patent_ids })))
  // Save session after streaming completes — the session_id now exists for
  // every conversation (纯 chat 首轮发送即建会话, 需求 2), so chat-only
  // conversations are persisted to the backend as well.
  const pendingSaveRef = useRef(false)
  useEffect(() => {
    if (streaming || messages.length === 0) return
    if (!sessionId) {
      // [sessdbg] Messages exist but no session yet — save would be lost.
      // Happens when createSession/restore has not resolved while a
      // follow-up already streamed (双 session 竞态排查日志).
      const last = messages[messages.length - 1]
      console.info(`[sessdbg] SAVE-SKIP-NOSID msgs=${messages.length} last=${last?.role} prefix=${JSON.stringify((last?.content || '').slice(0, 50))}`)
      return  // No session yet = no long task ever triggered
    }
    if (pendingSaveRef.current) return
    pendingSaveRef.current = true

    const timer = setTimeout(async () => {
      let toSave: { role: string; content: string; [key: string]: unknown }[] = []
      try {
        toSave = messages.map(m => ({
          role: m.role,
          content: m.content,
          ...(m.taskId ? { taskId: m.taskId } : {}),
          ...(m.resultSummary ? { resultSummary: m.resultSummary } : {}),
          ...(m.patent_ids ? { patent_ids: m.patent_ids } : {}),
          ...((m as any).results
            ? { results: pruneResultsForPersistence((m as any).results) }
            : {}),
        }))
        const last = toSave[toSave.length - 1]
        console.info(`[sessdbg] SAVE-EFFECT sid=${sessionId} msgs=${toSave.length} last=${last?.role} prefix=${JSON.stringify((last?.content || '').slice(0, 50))}`)
        await saveSessionMessages(sessionId, toSave)
        console.info(`[sessdbg] SAVE-EFFECT-OK sid=${sessionId} msgs=${toSave.length}`)
      } catch (e) {
        console.warn(`[sessdbg] SAVE-EFFECT-FAIL sid=${sessionId} msgs=${toSave.length}`, e)
      }
      pendingSaveRef.current = false
    }, 1000)

    return () => { clearTimeout(timer); pendingSaveRef.current = false }
  }, [streaming, messagesHash, sessionId])

  // 追问后立即刷新页面会丢失刚保存的消息 (保存有 1s 延迟) — 页面卸载前
  // (刷新/关闭) 用 fetch keepalive 尽力保存最新消息 (需求 2 补漏)。
  useEffect(() => {
    function saveOnUnload() {
      if (!sessionId || streaming || messages.length === 0) {
        // [sessdbg] Unload while save is blocked — streaming=true means the
        // 1s save effect also skipped, so the latest messages are lost.
        console.info(`[sessdbg] SAVE-PAGEHIDE-SKIP sid=${sessionId ?? 'null'} streaming=${streaming} msgs=${messages.length}`)
        return
      }
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
      const last = toSave[toSave.length - 1]
      console.info(`[sessdbg] SAVE-PAGEHIDE sid=${sessionId} msgs=${toSave.length} last=${last?.role} prefix=${JSON.stringify((last?.content || '').slice(0, 50))}`)
      try {
        // keepalive: 页面卸载期间请求仍会发出 (PUT 仅 fetch 支持,
        // sendBeacon 只支持 POST)
        void authHeaders().then((headers: Record<string, string>) => {
          void fetch(`${process.env.NEXT_PUBLIC_API_BASE || 'https://api.copiioai.com'}/session/${encodeURIComponent(sessionId)}/messages`, {
            method: 'PUT',
            headers,
            body: JSON.stringify({ messages: toSave, title: '' }),
            keepalive: true,
          }).catch(() => {})
        })
      } catch {
        // Non-critical — 1s 延迟保存仍会兜底 (若页面未真正卸载)
      }
    }
    window.addEventListener('pagehide', saveOnUnload)
    return () => window.removeEventListener('pagehide', saveOnUnload)
  }, [sessionId, streaming, messages])

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

  return (
    <div className="page active">
      <div className="chat-container">
        <div className="chat-messages" ref={chatContainerRef}>
          {showLanding ? (
            <ChatLanding
              input={input}
              setInput={setInput}
              streaming={streaming}
              send={send}
              abort={abort}
              selectedFiles={selectedFiles}
              addFiles={addFiles}
              removeFile={removeFile}
              setIsDragOver={setIsDragOver}
            />
          ) : (
            <>
              {messages.map((msg) => (
            <div key={msg.id} className={`chat-message-wrapper ${msg.role}`}>
              {msg.role === 'assistant' ? (
                <>
                  <MarkdownMessage
                    content={msg.content}
                    artifacts={msg.artifacts || []}
                    resultSummary={msg.resultSummary}
                    streaming={streaming && streamingId === msg.id}
                    statusSteps={streaming && streamingId === msg.id ? statusSteps : undefined}
                    statusElapsed={statusElapsed}
                    analysisType={(msg as any).analysisType}
                    tableColumns={(msg as any).tableColumns}
                    familyOverview={(msg as any).familyOverview}
                    jurisdictions={(msg as any).jurisdictions}
                    agentSteps={msg.agentSteps}
                    elapsedSeconds={msg.elapsedSeconds}
                    onRetry={handleRetryTask}
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
            </>
          )}
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
        {!showLanding && (
          <div className="chat-input-container">
            <ChatComposer
              input={input}
              setInput={setInput}
              streaming={streaming}
              send={send}
              abort={abort}
              selectedFiles={selectedFiles}
              addFiles={addFiles}
              removeFile={removeFile}
              setIsDragOver={setIsDragOver}
            />
          </div>
        )}
      </div>
    </div>
  )
}
