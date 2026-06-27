'use client'

import { useState, useRef, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import { queryStream, queryStreamWithFiles, getUserSceneStatus, getSceneKnowledge, pollLongTaskStatus, getLongTaskReportUrl, getSession, saveSessionMessages } from '@/services/api'
import { useI18n } from '@/lib/app-i18n'
import MarkdownMessage from '@/components/app/MarkdownMessage'
import { useChatSession } from '@/contexts/ChatContext'
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
  const { t } = useI18n()
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
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const isNearBottomRef = useRef(true)
  const [transientStatus, setTransientStatus] = useState('')
  const [enabledScenes, setEnabledScenes] = useState<any[]>([])
  const [sceneSmartQA, setSceneSmartQA] = useState<{name: string, desc: string}[]>([])
  const [sceneDeepResearch, setSceneDeepResearch] = useState<{name: string, desc: string}[]>([])

  useEffect(() => {
    getUserSceneStatus()
      .then(async (res) => {
        const subscribed = (res.scenes || []).filter((s: any) => s.subscribed)
        setEnabledScenes(subscribed)
        const smartQA: {name: string, desc: string}[] = []
        const deepResearch: {name: string, desc: string}[] = []
        for (const scene of subscribed) {
          try {
            const kr = await getSceneKnowledge(scene.id)
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
  }, [])

  // Load session from URL param (and resume long task polling if needed)
  const lastLoadedSidRef = useRef<string | null>(null)
  useEffect(() => {
    const sid = searchParams.get('session_id')
    if (!sid) {
      if (lastLoadedSidRef.current) {
        setMessages([])
        setSessionId(null)
        lastLoadedSidRef.current = null
        sessionLoadedRef.current = false
      }
      return
    }
    if (sid === lastLoadedSidRef.current) return

    lastLoadedSidRef.current = sid
    sessionLoadedRef.current = true

    ;(async () => {
      try {
        const data = await getSession(sid)
        const longTaskIds: string[] = data.long_task_ids || []
        if (data.messages && Array.isArray(data.messages)) {
          const loaded = data.messages
            .filter((m: { role: string; content: string }) => m.role && m.content)
            .map((m: { role: string; content: string }, i: number) => ({
              id: `hist_${i}_${Date.now()}`,
              role: m.role,
              content: m.content,
              artifacts: [],
            }))
          if (loaded.length > 0) {
            setMessages(loaded)
          }
        }
        setSessionId(sid)

        // Resume polling for any incomplete long tasks
        for (const tid of longTaskIds) {
          try {
            const status = await pollLongTaskStatus(tid)
            if (!status || status.status === 'unknown' || status.status === 'completed' || status.status === 'failed') {
              continue
            }
            // Task is still running — find the message that references it and resume
            const taskMsgId = `lt_resume_${tid}`
            setMessages(m => {
              const hasMsg = m.some(msg => msg.content.includes(tid))
              if (!hasMsg) {
                const phaseLabel = status.current_step || status.current_phase || ''
                const progress = status.progress != null ? `[${status.progress}%]` : ''
                return [...m, {
                  id: taskMsgId,
                  role: 'assistant',
                  content: progress ? `${progress} ${phaseLabel}` : `🔬 深度分析进行中... ${phaseLabel}`,
                  artifacts: [],
                }]
              }
              return m
            })
            startLongTaskPolling(tid, taskMsgId)
          } catch {
            // Task status check failed — skip
          }
        }
      } catch {
        // Session not found or error — start fresh
      }
    })()
  }, [searchParams, sessionId, setMessages, setSessionId])

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
    stopLongTaskPolling() // Stop any in-progress long task polling
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'

    const queryId = createChatId()
    setTransientStatus('')
    setMessages((m) => [...m, createChatMessage('user', text)])

    const assistant = createChatMessage('assistant', '')
    const assistantId = assistant.id
    setMessages((m) => [...m, assistant])
    setStreaming(true)
    setStreamingId(assistantId)

    // Collect full conversation history for context
    const conversationHistory = messages.map(m => ({
      role: m.role,
      content: m.content,
    }))

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const currentFiles = selectedFiles
      setSelectedFiles([])

      const body = currentFiles.length > 0
        ? await queryStreamWithFiles(text, queryId, controller.signal, currentFiles, conversationHistory)
        : await queryStream(text, queryId, controller.signal, conversationHistory)
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
              const taskId = String(event.task_id ?? '')
              const sid = String(event.session_id ?? '')
              setMessages((m) => updateAssistantMessage(m, assistantId,
                t('chat.longTaskCreated')
                  .replace('{taskId}', taskId)
                  .replace('{sessionId}', sid)
              ))
              // Use the backend-created session_id (don't create a new one)
              if (!sessionId && sid) {
                setSessionId(sid)
                const url = new URL(window.location.href)
                url.searchParams.set('session_id', sid)
                window.history.replaceState({}, '', url.toString())
                // Save current messages to the backend session
                const currentMsgs = messages.map(m => ({
                  role: m.role,
                  content: m.content,
                }))
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
      if ((err as Error).name !== 'AbortError') {
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId
              ? { ...msg, content: t('chat.queryFailed') }
              : msg
          )
        )
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
  const ALLOWED_EXTENSIONS = ['.pdf', '.docx']
  const ALLOWED_MIMES = [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
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
    return 'PDF'
  }

  function formatFileSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  function stopLongTaskPolling() {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }

  function startLongTaskPolling(taskId: string, assistantId: string) {
    stopLongTaskPolling()

    async function poll() {
      try {
        const data = await pollLongTaskStatus(taskId)
        if (!data || data.status === 'unknown') return

        const phaseLabel = data.current_step || data.current_phase || ''
        const progress = data.progress != null ? `[${data.progress}%]` : ''

        if (data.status === 'completed' || data.status === 'success') {
          stopLongTaskPolling()
          const files = (data.report_files || [])
            .map((f: { format: string }) => `[${f.format.toUpperCase()}](${getLongTaskReportUrl(taskId, f.format as 'pdf' | 'docx')})`)
            .join(' | ')
          setMessages((m) => replaceAssistantMessage(m, assistantId,
            t('chat.longTaskCompleted').replace('{files}', files)
          ))
        } else if (data.status === 'failed' || data.status === 'error') {
          stopLongTaskPolling()
          setMessages((m) => replaceAssistantMessage(m, assistantId,
            `${t('chat.longTaskFailed')} ${data.error_message || ''}`
          ))
        } else {
          setMessages((m) => replaceAssistantMessage(m, assistantId,
            t('chat.longTaskProgress')
              .replace('{progress}', progress)
              .replace('{phase}', phaseLabel)
          ))
        }
      } catch {
        // Non-fatal poll error; continue polling
      }
    }

    // Poll immediately, then every 3s
    poll()
    pollTimerRef.current = setInterval(poll, 3000)
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
                <span className="scene-hint-title">
                  ⚡ {t('chat.sceneHint')}
                </span>
              </div>
              <div className="scene-hint-scenes">
                {enabledScenes.map((scene, i) => (
                  <span key={i} className="scene-hint-scene-tag">
                    📦 {scene.name}
                  </span>
                ))}
              </div>

              {sceneSmartQA.length > 0 && (
                <div className="scene-hint-group scene-hint-group-smart">
                  <div className="scene-hint-group-header">
                    <span className="scene-hint-group-icon">💡</span>
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
                    <span className="scene-hint-group-icon">🔬</span>
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
              <span className="file-drop-hint">PDF, DOCX · Max 10 MB each · Up to 100 files</span>
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
              accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
              multiple
              onChange={e => { if (e.target.files) addFiles(e.target.files); e.target.value = '' }}
            />
            <button
              className="file-upload-btn"
              onClick={openFilePicker}
              aria-label="Attach patent files"
              title={t('chat.attachFiles') || 'Attach patent specification files (PDF, DOCX)'}
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
