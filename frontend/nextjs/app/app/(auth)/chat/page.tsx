'use client'

import { useState, useRef, useEffect } from 'react'
import { queryStream, getUserSceneStatus, getSceneKnowledge, pollLongTaskStatus, getLongTaskReportUrl } from '@/services/api'
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
  } = useChatSession()
  const bottomRef = useRef<HTMLDivElement | null>(null)
  const chatContainerRef = useRef<HTMLDivElement | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
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

    // Collect the last user + assistant exchange for long-task context
    const lastExchange: { role: string; content: string }[] = []
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'assistant' && lastExchange.length === 0) {
        lastExchange.unshift({ role: 'assistant', content: messages[i].content })
      } else if (messages[i].role === 'user' && lastExchange.length === 1) {
        lastExchange.unshift({ role: 'user', content: messages[i].content })
        break
      }
    }

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const body = await queryStream(text, queryId, controller.signal, lastExchange)
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
              setMessages((m) => updateAssistantMessage(m, assistantId,
                t('chat.longTaskCreated')
                  .replace('{taskId}', taskId)
                  .replace('{sessionId}', String(event.session_id ?? ''))
              ))
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

        <div className="chat-input-container">
          <div className="chat-input-wrapper">
            <textarea
              ref={textareaRef}
              className="chat-input"
              value={input}
              onChange={handleInput}
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
                onClick={send}
                disabled={!input.trim()}
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
