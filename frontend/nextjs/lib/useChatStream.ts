'use client'

import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { queryStream, queryStreamWithFiles, pollLongTaskBatchStatus, getLongTaskReportUrl, getSession, saveSessionMessages } from '@/services/api'
import { pollRecoverLongTask } from '@/lib/longTaskRecovery'
import { useI18n } from '@/lib/app-i18n'
import { useAuth } from '@/contexts/AuthContext'
import { useChatSession, type ChatMessage, type ChatStatusStep } from '@/contexts/ChatContext'
import { decodeArtifactChunksToResults, decodeResultsArtifact } from '@/lib/chatSession'
import { persistResultsSetToStorage } from '@/lib/resultsStore'
import { persistChatToStorage } from '@/lib/chatStore'
import { resultsPath } from '@/lib/appRoutes'
import {
  addAssistantArtifactComplete,
  addAssistantArtifactEnd,
  addAssistantArtifactStart,
  addAssistantPatentIds,
  applyAgentElapsed,
  applyAgentObservation,
  applyAgentStep,
  createChatId,
  createChatMessage,
  updateAssistantMessage,
  replaceAssistantMessage,
} from '@/lib/chatSession'

function cleanGarbledText(text: string): string {
  if (!text) return text
  return text
    .replace(/�/g, '')
    .replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])/g, '')
    .replace(/(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, '')
    .replace(/[-]/g, '')
}

export function useChatStream() {
  const { t, lang } = useI18n()
  const { user, requireAuth } = useAuth()
  const router = useRouter()
  const {
    messages, setMessages, input, setInput,
    streaming, setStreaming, streamingId, setStreamingId,
    abortRef, sessionId, setSessionId,
    statusSteps, setStatusSteps, statusElapsed, setStatusElapsed,
  } = useChatSession()

  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  // Batch polling: one global timer → one POST /batch_status for all active tasks
  const activeTasksRef = useRef<Map<string, string>>(new Map())       // taskId → assistantId
  const globalPollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const longTaskReceivedRef = useRef(false)
  // Artifact chunks are buffered in a ref and committed in ONE state update
  // at artifact_end.  Per-chunk setMessages rebuilt the artifacts array —
  // and re-persisted the whole conversation (multi-MB CSV/XLSX included) —
  // for every 32KB chunk, freezing the tab for minutes on 100-row results.
  const artifactChunksRef = useRef<Map<string, string[]>>(new Map())  // artifactId → chunks

  const MAX_FILE_SIZE = 10 * 1024 * 1024 // 10 MB
  const MAX_FILE_COUNT = 100
  const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.xml']
  const ALLOWED_MIMES = [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/xml',
    'text/xml',
  ]

  function addFiles(files: FileList | File[]) {
    const incoming = Array.from(files)
    const valid: File[] = []
    for (const f of incoming) {
      const ext = '.' + f.name.split('.').pop()?.toLowerCase()
      if (!ALLOWED_EXTENSIONS.includes(ext) && !ALLOWED_MIMES.includes(f.type)) {
        // Silently skip unsupported files
        continue
      }
      if (f.size > MAX_FILE_SIZE) continue
      if (f.size < 50) continue
      valid.push(f)
    }
    setSelectedFiles(prev => {
      const merged = [...prev, ...valid].slice(0, MAX_FILE_COUNT)
      return merged
    })
  }

  function removeFile(index: number) {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index))
  }

  // Keep a ref to the latest send() so the pending auth callback always
  // invokes the current render's closure (with up-to-date `user`), not a
  // stale one from the anonymous render where `user` was null.
  const sendRef = useRef<(presetText?: string) => Promise<void>>(async () => {})
  sendRef.current = send

  // Snapshot of the message list for the pre-navigation persistence in
  // the stream finally block.  A layout effect (not a passive one) keeps
  // it fresh as soon as each commit lands; the finally block yields a
  // task before reading it so the final streaming update is committed.
  const messagesRef = useRef<ChatMessage[]>(messages)
  useLayoutEffect(() => {
    messagesRef.current = messages
  })

  async function send(presetText: string = '') {
    const text = (presetText || input).trim()
    if (!text || streaming) return

    if (!user) {
      requireAuth(() => sendRef.current(presetText), lang === 'en' ? 'Sign in to get your answer' : '登录后立即获得答案')
      return
    }

    setInput('')
    // NOTE: the chat page previously reset the auto-growing textarea height
    // here (textareaRef owned by the page). The hook cannot reach the page's
    // DOM ref; the page compensates by resetting the textarea height whenever
    // input becomes empty (see page.tsx). Behavior preserved.

    const queryId = createChatId()
    setStatusSteps([])
    setStatusElapsed(0)
    longTaskReceivedRef.current = false

    const userMsg = createChatMessage('user', text)
    const assistant = createChatMessage('assistant', '')
    const assistantId = assistant.id

    // Local (synchronous) tracking of the JSON artifact so the post-stream
    // navigation decision never depends on React state commit timing.
    let pendingJsonId: string | null = null
    let pendingJsonChunks: string[] = []
    let decodedSetId: string | null = null

    // Preserve all long task cards (running / completed / failed) so the
    // user can see multiple concurrent or queued tasks in one conversation.
    setMessages((m) => [...m, userMsg, assistant])
    setStreaming(true)
    setStreamingId(assistantId)

    // Collect full conversation history for context — include the new messages
    const conversationHistory = ([
      ...messages,
      userMsg,
      assistant,
    ] as ChatMessage[]).map(m => ({
      role: m.role,
      content: m.content,
      ...(m.patent_ids ? { patent_ids: m.patent_ids } : {}),
    }))

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const currentFiles = selectedFiles
      setSelectedFiles([])

      const body = currentFiles.length > 0
        ? await queryStreamWithFiles(text, queryId, controller.signal, currentFiles, conversationHistory, sessionId || undefined)
        : await queryStream(text, queryId, controller.signal, conversationHistory, sessionId || undefined)
      const reader = body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const raw = line.slice(5).trim()
          if (raw === '[DONE]') continue
          let evt: unknown
          try {
            evt = JSON.parse(raw)
          } catch {
            // non-JSON line, ignore
            continue
          }

          if (evt && typeof evt === 'object') {
            const event = evt as Record<string, unknown>
            if (event.type === 'status') {
              const msg = cleanGarbledText(String(event.message ?? '')).trim()
              if (msg) {
                setStatusSteps((steps) => {
                  const last = steps[steps.length - 1]
                  if (last && last.state === 'running' && last.message === msg) {
                    return steps
                  }
                  // A new status means the previous step finished: mark
                  // any running step done so only the newest one carries
                  // the elapsed timer.
                  return [
                    ...steps.map((s) => (s.state === 'running' ? { ...s, state: 'done' as const } : s)),
                    { id: Date.now(), message: msg, state: 'running' },
                  ]
                })
                setStatusElapsed(0)
              }
              continue
            }
            if (event.type === 'artifact_start') {
              // Track the JSON artifact's chunks locally so navigation can be
              // decided synchronously after streaming — independent of React
              // state commit timing.
              if (event.format === 'json') {
                pendingJsonId = String(event.artifact_id ?? event.artifactId ?? '')
                pendingJsonChunks = []
              }
              artifactChunksRef.current.set(
                String(event.artifact_id ?? event.artifactId ?? ''), [])
              setMessages((m) => addAssistantArtifactStart(m, assistantId, event))
              continue
            }
            if (event.type === 'artifact_chunk') {
              const chunkArtifactId = String(event.artifact_id ?? event.artifactId ?? '')
              if (pendingJsonId !== null && chunkArtifactId === pendingJsonId) {
                pendingJsonChunks.push(String(event.data ?? ''))
              }
              // Buffer in the ref — NO state update per chunk (see
              // artifactChunksRef comment).
              const buffer = artifactChunksRef.current.get(chunkArtifactId)
              if (buffer) buffer.push(String(event.data ?? ''))
              continue
            }
            if (event.type === 'artifact_end') {
              const endArtifactId = String(event.artifact_id ?? event.artifactId ?? '')
              const bufferedChunks = artifactChunksRef.current.get(endArtifactId) ?? []
              if (pendingJsonId !== null && endArtifactId === pendingJsonId) {
                const decodedResults = decodeArtifactChunksToResults(
                  pendingJsonChunks, pendingJsonId,
                )
                decodedSetId = decodedResults?.setId ?? null
                pendingJsonId = null
                // Persist a pruned copy to browser localStorage so the
                // results survive refresh / tab reopen.  Unavailable
                // storage degrades silently (no-op).
                if (decodedResults) {
                  persistResultsSetToStorage(
                    window.localStorage,
                    decodedResults,
                    { sessionId, queryText: text, savedAt: Date.now() },
                  )
                }
              }
              // Commit the buffered chunks in a single state update and mark
              // the artifact complete — one write instead of one per chunk.
              setMessages((m) => addAssistantArtifactComplete(
                m,
                assistantId,
                endArtifactId,
                bufferedChunks,
              ))
              artifactChunksRef.current.delete(endArtifactId)
              // Decode the completed JSON results artifact into message.results
              // for the results page. Idempotent — safe to call unconditionally.
              setMessages((m) => decodeResultsArtifact(m, assistantId))
              continue
            }
            if (event.type === 'patent_ids') {
              const ids = event.patent_ids
              if (Array.isArray(ids) && ids.length > 0) {
                setMessages((m) => addAssistantPatentIds(m, assistantId, ids))
              }
              continue
            }
            if (event.type === 'long_task_created') {
              longTaskReceivedRef.current = true
              const taskId = String(event.task_id ?? '')
              const sid = String(event.session_id ?? '')
              const isQueued = String(event.status ?? '') === 'queued'
              const initContent = (isQueued
                ? t('chat.longTaskQueuedWithId').replace('{taskId}', taskId)
                : t('chat.longTaskProgress')
                    .replace('{progress}', '[0%]')
                    .replace('{phase}', t('chat.phasePreparing'))
              ) + ` Task ID: ${taskId}`
              setMessages((m) => {
                // Dedup: remove any stale task messages with the same taskId
                const cleaned = m.filter((msg: { taskId?: string }) => msg.taskId !== taskId)
                const updated = replaceAssistantMessage(cleaned, assistantId, initContent)
                return updated.map((msg: { id: string; [key: string]: unknown }) =>
                  msg.id === assistantId ? { ...msg, taskId } : msg
                )
              })
              // Use the backend-created session_id (don't create a new one)
              if (!sessionId && sid) {
                setSessionId(sid)
                const url = new URL(window.location.href)
                url.searchParams.set('session_id', sid)
                window.history.replaceState({}, '', url.toString())
                const currentMsgs = [
                  ...messages,
                  { role: userMsg.role, content: userMsg.content },
                  { role: assistant.role, content: initContent, taskId },
                ]
                saveSessionMessages(sid, currentMsgs).catch(() => {})
              }
              // Start polling for progress updates
              startLongTaskPolling(taskId, assistantId)
              continue
            }
            if (event.type === 'step') {
              setMessages((m) => applyAgentStep(m, assistantId, event))
              continue
            }
            if (event.type === 'observation') {
              setMessages((m) => applyAgentObservation(m, assistantId, event))
              continue
            }
            if (event.type === 'agent_elapsed') {
              setMessages((m) => applyAgentElapsed(m, assistantId, event))
              continue
            }
          }
          if (evt && typeof evt === 'object' && 'error' in evt && evt.error) {
            throw new Error(String(evt.error))
          }

          const token = typeof evt === 'string'
            ? evt
            : (
              evt && typeof evt === 'object'
                ? (
                  ('content' in evt ? evt.content : undefined) ??
                  ('token' in evt ? evt.token : undefined) ??
                  ('answer' in evt ? evt.answer : undefined) ??
                  ''
                )
                : ''
            )
          if (token) {
            setStatusSteps([])
            setMessages((m) => updateAssistantMessage(m, assistantId, cleanGarbledText(String(token))))
          }
        }
      }
    } catch (err) {
      if ((err as Error).name !== 'AbortError' && !longTaskReceivedRef.current) {
        // SSE may have timed out after the backend already created a long task
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId
              ? { ...msg, content: t('chat.queryRecovering') }
              : msg
          )
        )

        const recovered = await pollRecoverLongTask(queryId)
        if (recovered) {
          longTaskReceivedRef.current = true
          const isQueued = recovered.status === 'queued'
          const initContent = (isQueued
            ? t('chat.longTaskQueuedWithId').replace('{taskId}', recovered.taskId)
            : t('chat.longTaskProgress')
                .replace('{progress}', '[0%]')
                .replace('{phase}', t('chat.phasePreparing'))
          ) + ` Task ID: ${recovered.taskId}`

          setMessages((m) => {
            const cleaned = m.filter((msg) => msg.taskId !== recovered.taskId)
            const updated = replaceAssistantMessage(cleaned, assistantId, initContent) as ChatMessage[]
            return updated.map((msg: ChatMessage) =>
              msg.id === assistantId ? { ...msg, taskId: recovered.taskId } : msg
            )
          })

          const sid = recovered.sessionId
          if (!sessionId && sid) {
            setSessionId(sid)
            const url = new URL(window.location.href)
            url.searchParams.set('session_id', sid)
            window.history.replaceState({}, '', url.toString())
            saveSessionMessages(sid, [
              { role: userMsg.role, content: userMsg.content },
              { role: assistant.role, content: initContent },
            ]).catch(() => {})
          }

          startLongTaskPolling(recovered.taskId, assistantId)
        } else {
          setMessages((m) =>
            m.map((msg) =>
              msg.id === assistantId
                ? { ...msg, content: t('chat.queryFailedWithHint') }
                : msg
            )
          )
        }
      }
    } finally {
      setStatusSteps([])
      setStatusElapsed(0)
      setStreaming(false)
      setStreamingId(null)
      abortRef.current = null

      // Auto-open the results page once a search has streamed a decoded
      // results set — no intermediate card click required.  decodedSetId is
      // assigned synchronously in the SSE loop (independent of React commit
      // timing), so this check is deterministic.
      if (decodedSetId) {
        // Yield a task so React commits the final streaming update (and
        // its layout effects) before the snapshot is persisted.  Then
        // persist the conversation synchronously before navigating: the
        // results page mounts a fresh ChatProvider (separate instance
        // from the landing page's) and hydrates it from this copy.
        await new Promise((resolve) => setTimeout(resolve, 0))
        persistChatToStorage(window.sessionStorage, messagesRef.current)
        router.push(resultsPath(decodedSetId, sessionId))
      }
    }
  }

  function stopLongTaskPolling(taskId?: string) {
    if (taskId) {
      activeTasksRef.current.delete(taskId)
      // If no more active tasks, stop the global poll timer
      if (activeTasksRef.current.size === 0 && globalPollTimerRef.current) {
        clearInterval(globalPollTimerRef.current)
        globalPollTimerRef.current = null
      }
    } else {
      // Stop all polling
      activeTasksRef.current.clear()
      if (globalPollTimerRef.current) {
        clearInterval(globalPollTimerRef.current)
        globalPollTimerRef.current = null
      }
    }
  }

  // Shared batch poll loop — fires one POST /batch_status for all active tasks
  function ensureGlobalPollLoop() {
    if (globalPollTimerRef.current) return // already running

    async function pollAll() {
      const activeIds = Array.from(activeTasksRef.current.keys())
      if (activeIds.length === 0) {
        // Nothing to poll — stop the loop
        if (globalPollTimerRef.current) {
          clearInterval(globalPollTimerRef.current)
          globalPollTimerRef.current = null
        }
        return
      }

      try {
        const batch = await pollLongTaskBatchStatus(activeIds)

        for (const [taskId, assistantId] of activeTasksRef.current) {
          const data = batch[taskId]
          if (!data || data.status === 'unknown') {
            setMessages((m) => m.map(msg =>
              msg.taskId === taskId
                ? { ...msg, content: t('chat.longTaskProgress')
                    .replace('{progress}', '[0%]')
                    .replace('{phase}', t('chat.phasePreparing'))
                    + ` Task ID: ${taskId}` }
                : msg
            ))
            continue
          }

          if (data.status === 'queued') {
            setMessages((m) => m.map(msg =>
              msg.taskId === taskId
                ? { ...msg, content: t('chat.longTaskQueuedWithId').replace('{taskId}', taskId) }
                : msg
            ))
            continue
          }

          const phaseLabel = data.current_step || data.current_phase || ''
          const progress = data.progress != null ? `[${data.progress}%]` : ''

          function findAndUpdate(messages: any[], newContent: string, summary?: string, extraFields?: Record<string, any>) {
            const idx = messages.findIndex(msg => msg.taskId === taskId)
            if (idx >= 0) {
              return messages.map((msg, i) =>
                i === idx
                  ? {
                      ...msg,
                      content: newContent,
                      resultSummary: summary ?? msg.resultSummary,
                      ...(extraFields || {}),
                    }
                  : msg
              )
            }
            // Fallback: try assistantId (new task — clear stale metadata from
            // previous tasks so family labels don't leak into batch analysis etc.)
            return messages.map(msg =>
              msg.id === assistantId
                ? {
                    ...msg,
                    content: newContent,
                    resultSummary: summary ?? msg.resultSummary,
                    analysisType: undefined,
                    jurisdictions: undefined,
                    familyOverview: undefined,
                    tableColumns: undefined,
                    ...(extraFields || {}),
                  }
                : msg
            )
          }

          if (data.status === 'completed' || data.status === 'success') {
            stopLongTaskPolling(taskId)
            const files = (data.report_files || [])
              .map((f: { format: string }) => `[${f.format.toUpperCase()}](${getLongTaskReportUrl(taskId, f.format as 'pdf' | 'docx')})`)
              .join(' | ')
            // Preserve patent_ids from completed task status so follow-up
            // conversation_refs queries can find them in conversation_history.
            const taskPatentIds: string[] | undefined =
              Array.isArray(data.patent_ids) ? data.patent_ids as string[] : undefined
            setMessages((m) => {
              const updated = findAndUpdate(
                m,
                t('chat.longTaskCompleted').replace('{files}', files),
                data.result_summary,
              )
              if (taskPatentIds && taskPatentIds.length > 0) {
                return updated.map(msg =>
                  msg.taskId === taskId
                    ? { ...msg, patent_ids: taskPatentIds }
                    : msg
                )
              }
              return updated
            })
          } else if (data.status === 'paused') {
            // Don't stop polling — the task may be resumed later
            const pausedLabel = data.current_step || `已暂停（进度 ${data.progress || 0}%）`
            setMessages((m) => findAndUpdate(
              m,
              `⏸ ${pausedLabel} Task ID: ${taskId}`,
              data.result_summary,
            ))
          } else if (data.status === 'cancelling') {
            // Backend is processing the stop request — show progress until cancelled
            const pct = data.progress != null ? `[${data.progress}%]` : ''
            setMessages((m) => findAndUpdate(
              m,
              `⏹ ${t('chat.longTaskStopping').replace('{pct}', pct).replace('{taskId}', taskId)}`,
              data.result_summary,
            ))
          } else if (data.status === 'cancelled') {
            stopLongTaskPolling(taskId)
            setMessages((m) => findAndUpdate(m, `⏹ Task cancelled Task ID: ${taskId}`))
          } else if (data.status === 'failed' || data.status === 'error') {
            stopLongTaskPolling(taskId)
            setMessages((m) => findAndUpdate(
              m,
              `${t('chat.longTaskFailed')} ${data.error_message || ''}`,
              data.result_summary,
            ))
          } else {
            const newContent = t('chat.longTaskProgress')
              .replace('{progress}', progress)
              .replace('{phase}', phaseLabel)
              + ` Task ID: ${taskId}`
            const extraFields: Record<string, any> = {}
            if (data.analysis_type) extraFields.analysisType = data.analysis_type
            if (data.table_columns) extraFields.tableColumns = data.table_columns
            if (data.family_overview) extraFields.familyOverview = data.family_overview
            if (data.jurisdictions) extraFields.jurisdictions = data.jurisdictions
            setMessages((m) => {
              // Preserve previously-set fields when backend doesn't resend them,
              // BUT only from the SAME task — never leak metadata across tasks.
              const sameTask = m.find(msg => msg.taskId === taskId) as any
              if (!extraFields.analysisType && sameTask?.analysisType) extraFields.analysisType = sameTask.analysisType
              if (!extraFields.tableColumns && sameTask?.tableColumns) extraFields.tableColumns = sameTask.tableColumns
              if (!extraFields.familyOverview && sameTask?.familyOverview) extraFields.familyOverview = sameTask.familyOverview
              if (!extraFields.jurisdictions && sameTask?.jurisdictions) extraFields.jurisdictions = sameTask.jurisdictions
              return findAndUpdate(m, newContent, data.result_summary, extraFields)
            })
          }
        }
      } catch {
        // Non-fatal batch poll error; continue polling
      }
    }

    // Poll immediately, then every 1.5s (faster during report summary streaming)
    pollAll()
    globalPollTimerRef.current = setInterval(pollAll, 1500)
  }

  function startLongTaskPolling(taskId: string, assistantId: string) {
    // Register this task in the active set
    activeTasksRef.current.set(taskId, assistantId)
    // Ensure the single global poll loop is running
    ensureGlobalPollLoop()
  }

  // Cleanup poll timer on unmount
  useEffect(() => {
    return () => {
      stopLongTaskPolling()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- stopLongTaskPolling recreated per render
  }, [])

  // Live elapsed-seconds counter while a status step is running
  useEffect(() => {
    if (!streaming) return
    const id = setInterval(() => setStatusElapsed((s) => s + 1), 1000)
    return () => clearInterval(id)
  }, [streaming])

  return {
    send: (files: File[] = [], presetText: string = '') => {
      if (files.length > 0) setSelectedFiles((prev) => [...prev, ...files])
      return sendRef.current(presetText)
    },
    abort: () => abortRef.current?.abort(),
    statusSteps,
    statusElapsed,
    selectedFiles,
    setSelectedFiles,
    addFiles,
    removeFile,
    isDragOver,
    setIsDragOver,
    // Exposed so the chat page's session-load effect can stop all polling
    // when navigating to a URL without session_id, and resume polling for
    // long tasks loaded from a saved session. These operate on the hook-owned
    // activeTasksRef/globalPollTimerRef.
    stopLongTaskPolling,
    startLongTaskPolling,
  }
}
