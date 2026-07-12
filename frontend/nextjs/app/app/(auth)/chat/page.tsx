'use client'

import { useState, useRef, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import { queryStream, queryStreamWithFiles, getUserSceneStatus, getSceneKnowledge, pollLongTaskBatchStatus, getLongTaskReportUrl, getSession, saveSessionMessages } from '@/services/api'
import { pollRecoverLongTask } from '@/lib/longTaskRecovery'
import { useI18n } from '@/lib/app-i18n'
import MarkdownMessage from '@/components/app/MarkdownMessage'
import { useChatSession } from '@/contexts/ChatContext'
import type { ChatMessage } from '@/contexts/ChatContext'
import { copyTextToClipboard } from '@/lib/clipboard'
import {
  addAssistantArtifactChunk,
  addAssistantArtifactEnd,
  addAssistantArtifactStart,
  createChatId,
  createChatMessage,
  updateAssistantMessage,
  replaceAssistantMessage,
} from '@/lib/chatSession'

function UserCopyButton({ content }: { content: string }) {
  const { t } = useI18n()
  const [copied, setCopied] = useState(false)
  async function handleCopy() {
    if (await copyTextToClipboard(content)) {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }
  return (
    <button
      className={`user-copy-button${copied ? ' copied' : ''}`}
      onClick={handleCopy}
      data-tooltip={t('chat.copy')}
      data-copied-tooltip={t('chat.copied')}
      aria-label={t('chat.copyContent')}
    >
      {copied ? (
        <svg viewBox="0 0 24 24" fill="none" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
      )}
    </button>
  )
}

export default function Chat() {
  const { t, lang } = useI18n()
  const {
    messages,
    setMessages,
    input,
    setInput,
    streaming,
    setStreaming,
    streamingId,
    setStreamingId,
    abortRef,
    sessionId,
    setSessionId,
  } = useChatSession()
  const searchParams = useSearchParams()
  const sessionLoadedRef = useRef(false)
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const chatContainerRef = useRef<HTMLDivElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  // Batch polling: one global timer 鈫?one POST /batch_status for all active tasks
  const activeTasksRef = useRef<Map<string, string>>(new Map())       // taskId 鈫?assistantId
  const globalPollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const longTaskReceivedRef = useRef(false)
  const isNearBottomRef = useRef(true)
  const [transientStatus, setTransientStatus] = useState('')
  const [enabledScenes, setEnabledScenes] = useState<any[]>([])
  const [sceneSmartQA, setSceneSmartQA] = useState<{name: string, desc: string}[]>([])
  const [sceneDeepResearch, setSceneDeepResearch] = useState<{name: string, desc: string}[]>([])

  useEffect(() => {
    getUserSceneStatus(lang)
      .then(async (res) => {
        const subscribed = (res.scenes || []).filter((s: any) => s.subscribed)
        setEnabledScenes(subscribed)
        const smartQA: {name: string, desc: string}[] = []
        const deepResearch: {name: string, desc: string}[] = []
        for (const scene of subscribed) {
          try {
            const kr = await getSceneKnowledge(scene.id, lang)
            const items = kr.knowledge || []
            items.forEach((item: any) => {
              const example = {
                name: scene.name,
                desc: item.description || item.question,
              }
              if (item.type === 3) {
                deepResearch.push(example)
              } else {
                smartQA.push(example)
              }
            })
          } catch {}
        }
        setSceneSmartQA(smartQA)
        setSceneDeepResearch(deepResearch)
      })
      .catch(() => {})
  }, [lang])

  // Load session from URL param (and resume long task polling if needed)
  const lastLoadedSidRef = useRef<string | null>(null)
  useEffect(() => {
    const sid = searchParams.get('session_id')
    if (!sid) {
      // Always clear when navigating to a URL without session_id (e.g. 鏂板璇?.
      // This handles both the case where the component persisted across
      // a client-side navigation and the case where it freshly mounted
      // with stale ChatProvider state from the parent layout.
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
            .map((m: { role: string; content: string; taskId?: string }, i: number) => ({
              id: `hist_${i}_${Date.now()}`,
              role: m.role,
              content: m.content,
              taskId: (m as any).taskId || undefined,
              artifacts: [],
            }))
            // Strip orphan long-task messages (馃敩/✅❌without taskId).
            // These were saved before taskId was attached during SSE.
            // The resume loop below will recreate them with proper taskId,
            // avoiding duplicates that never update.
            .filter((m: { taskId?: string; content: string }) =>
              m.taskId || (!m.content.includes('馃敩') && !m.content.includes('✅) && !m.content.includes('❌))
            )
          if (loaded.length > 0) {
            setMessages(loaded)
            // Scroll to bottom after loading session messages
            requestAnimationFrame(() => {
              isNearBottomRef.current = true
              bottomRef.current?.scrollIntoView({ behavior: 'instant' as ScrollBehavior })
            })
          }
        }
        setSessionId(sid)

        // Resume polling for any incomplete long tasks 鈥?batch fetch all statuses
        if (longTaskIds.length > 0) {
          try {
            const batch = await pollLongTaskBatchStatus(longTaskIds)
            for (const tid of longTaskIds) {
              const status = batch[tid]
              if (!status) continue

            // Session save happens ~1s after SSE end, but the task may complete
            // minutes later.  The in-memory message transitions to ✅❌via
            // polling, but the saved session still has the stale 馃敩 content.
            // Update completed/failed messages so the card shows the final state
            // and so send()'s filter (which checks for ✅❌ preserves them.
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
              : '馃敩 娣卞害鍒嗘瀽杩涜涓?..') + ` 浠诲姟ID: ${tid}`
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
            // Batch status fetch failed 鈥?skip resume
          }
        }
      } catch {
        // Session not found or error 鈥?start fresh
      }
    })()

    return () => { cancelled = true }
  }, [searchParams, sessionId, setMessages, setSessionId])

  // Save session after streaming completes 鈥?but ONLY if a session already exists
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
        }))
        await saveSessionMessages(sessionId, toSave)
      } catch {
        // Non-critical
      }
      pendingSaveRef.current = false
    }, 1000)

    return () => { clearTimeout(timer); pendingSaveRef.current = false }
  }, [streaming, messages.length, sessionId])

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

  async function send() {
    const text = input.trim()
    if (!text || streaming) return
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    const queryId = createChatId()
    setTransientStatus('')
    longTaskReceivedRef.current = false

    const userMsg = createChatMessage('user', text)
    const assistant = createChatMessage('assistant', '')
    const assistantId = assistant.id

    // Preserve all long task cards (running / completed / failed) so the
    // user can see multiple concurrent or queued tasks in one conversation.
    setMessages((m) => [...m, userMsg, assistant])
    setStreaming(true)
    setStreamingId(assistantId)

    // Collect full conversation history for context 鈥?include the new messages
    const conversationHistory = [
      ...messages,
      userMsg,
      assistant,
    ].map(m => ({
      role: m.role,
      content: m.content,
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
              setTransientStatus(String(event.message ?? ''))
              continue
            }
            if (event.type === 'artifact_start') {
              setMessages((m) => addAssistantArtifactStart(m, assistantId, event))
              continue
            }
            if (event.type === 'artifact_chunk') {
              setMessages((m) => addAssistantArtifactChunk(
                m,
                assistantId,
                String(event.artifact_id ?? event.artifactId ?? ''),
                String(event.data ?? '')
              ))
              continue
            }
            if (event.type === 'artifact_end') {
              setMessages((m) => addAssistantArtifactEnd(
                m,
                assistantId,
                String(event.artifact_id ?? event.artifactId ?? '')
              ))
              continue
            }
            if (event.type === 'long_task_created') {
              longTaskReceivedRef.current = true
              const taskId = String(event.task_id ?? '')
              const sid = String(event.session_id ?? '')
              const isQueued = String(event.status ?? '') === 'queued'
              const initContent = (isQueued
                ? '馃敩 娣卞害鍒嗘瀽宸叉帓闃燂紝灏嗗湪褰撳墠浠诲姟瀹屾垚鍚庤嚜鍔ㄥ紑濮?..'
                : t('chat.longTaskProgress')
                    .replace('{progress}', '[0%]')
                    .replace('{phase}', '姝ｅ湪鍑嗗涓撳埄鍒嗘瀽...')
              ) + ` 浠诲姟ID: ${taskId}`
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
            setTransientStatus('')
            setMessages((m) => updateAssistantMessage(m, assistantId, String(token)))
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
            ? '馃敩 娣卞害鍒嗘瀽宸叉帓闃燂紝灏嗗湪褰撳墠浠诲姟瀹屾垚鍚庤嚜鍔ㄥ紑濮?..'
            : t('chat.longTaskProgress')
                .replace('{progress}', '[0%]')
                .replace('{phase}', '姝ｅ湪鍑嗗涓撳埄鍒嗘瀽...')
          ) + ` 浠诲姟ID: ${recovered.taskId}`

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
      setTransientStatus('')
      setStreaming(false)
      setStreamingId(null)
      abortRef.current = null
    }
  }

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

  function abort() {
    abortRef.current?.abort()
  }

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

  // Shared batch poll loop 鈥?fires one POST /batch_status for all active tasks
  function ensureGlobalPollLoop() {
    if (globalPollTimerRef.current) return // already running

    async function pollAll() {
      const activeIds = Array.from(activeTasksRef.current.keys())
      if (activeIds.length === 0) {
        // Nothing to poll 鈥?stop the loop
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
                    .replace('{phase}', '姝ｅ湪鍑嗗涓撳埄鍒嗘瀽...')
                    + ` 浠诲姟ID: ${taskId}` }
                : msg
            ))
            continue
          }

          if (data.status === 'queued') {
            setMessages((m) => m.map(msg =>
              msg.taskId === taskId
                ? { ...msg, content: `馃敩 娣卞害鍒嗘瀽鎺掗槦涓紝灏嗗湪褰撳墠浠诲姟瀹屾垚鍚庤嚜鍔ㄥ紑濮?.. 浠诲姟ID: ${taskId}` }
                : msg
            ))
            continue
          }

          const phaseLabel = data.current_step || data.current_phase || ''
          const progress = data.progress != null ? `[${data.progress}%]` : ''

          function findAndUpdate(messages: any[], newContent: string, summary?: string) {
            const idx = messages.findIndex(msg => msg.taskId === taskId)
            if (idx >= 0) {
              return messages.map((msg, i) =>
                i === idx
                  ? {
                      ...msg,
                      content: newContent,
                      resultSummary: summary ?? msg.resultSummary,
                    }
                  : msg
              )
            }
            // Fallback: try assistantId
            return messages.map(msg =>
              msg.id === assistantId
                ? {
                    ...msg,
                    content: newContent,
                    resultSummary: summary ?? msg.resultSummary,
                  }
                : msg
            )
          }

          if (data.status === 'completed' || data.status === 'success') {
            stopLongTaskPolling(taskId)
            const files = (data.report_files || [])
              .map((f: { format: string }) => `[${f.format.toUpperCase()}](${getLongTaskReportUrl(taskId, f.format as 'pdf' | 'docx')})`)
              .join(' | ')
            setMessages((m) => findAndUpdate(
              m,
              t('chat.longTaskCompleted').replace('{files}', files),
              data.result_summary,
            ))
          } else if (data.status === 'paused') {
            // Don't stop polling 鈥?the task may be resumed later
            const pausedLabel = data.current_step || `宸叉殏鍋滐紙杩涘害 ${data.progress || 0}%锛塦
            setMessages((m) => findAndUpdate(
              m,
              `鈴?${pausedLabel} 浠诲姟ID: ${taskId}`,
              data.result_summary,
            ))
          } else if (data.status === 'cancelling') {
            // Backend is processing the stop request 鈥?show progress until cancelled
            const pct = data.progress != null ? `[${data.progress}%]` : ''
            setMessages((m) => findAndUpdate(
              m,
              `鈴?姝ｅ湪鍋滄... ${pct} 浠诲姟ID: ${taskId}`,
              data.result_summary,
            ))
          } else if (data.status === 'cancelled') {
            stopLongTaskPolling(taskId)
            setMessages((m) => findAndUpdate(m, `鈴?浠诲姟宸插彇娑?浠诲姟ID: ${taskId}`))
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
              + ` 浠诲姟ID: ${taskId}`
            setMessages((m) => findAndUpdate(m, newContent, data.result_summary))
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
    return () => stopLongTaskPolling()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

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
          {enabledScenes.length > 0 && (
            <div className="scene-hint scene-hint-persistent">
              <div className="scene-hint-header">
                <span className="scene-hint-title">{t('chat.sceneHint')}</span>
              </div>
              <div className="scene-hint-scenes">
                {enabledScenes.map((scene, i) => (
                  <span key={i} className="scene-hint-scene-tag">{scene.name}</span>
                ))}
              </div>

              {sceneSmartQA.length > 0 && (
                <div className="scene-hint-group scene-hint-group-smart">
                  <div className="scene-hint-group-header">
                    <span className="scene-hint-group-label">{t('chat.sceneSmartQA')}</span>
                  </div>
                  <ul className="scene-hint-list">
                    {sceneSmartQA.map((ex, i) => (
                      <li key={i} className="scene-hint-item">
                        {ex.desc}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {sceneDeepResearch.length > 0 && (
                <div className="scene-hint-group scene-hint-group-deep">
                  <div className="scene-hint-group-header">
                    <span className="scene-hint-group-label">{t('chat.sceneDeepResearch')}</span>
                  </div>
                  <ul className="scene-hint-list">
                    {sceneDeepResearch.map((ex, i) => (
                      <li key={i} className="scene-hint-item">
                        {ex.desc}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
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
              <span className="file-drop-hint">PDF, DOCX, XML 路 Max 10 MB each 路 Up to 100 files</span>
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
                onClick={send}
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
