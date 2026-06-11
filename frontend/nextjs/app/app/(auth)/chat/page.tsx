'use client'

import { useState, useRef, useEffect } from 'react'
import { queryStream, getUserSceneStatus, getSceneKnowledge } from '@/services/api'
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
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const [transientStatus, setTransientStatus] = useState('')
  const [sceneHints, setSceneHints] = useState<string[]>([])
  const [sceneHintVisible, setSceneHintVisible] = useState(true)

  useEffect(() => {
    getUserSceneStatus()
      .then(async (res) => {
        const subscribed = (res.scenes || []).filter((s: any) => s.subscribed)
        const hints: string[] = []
        for (const scene of subscribed) {
          try {
            const kr = await getSceneKnowledge(scene.id)
            const items = kr.knowledge || []
            items.slice(0, 3).forEach((item: any) => {
              hints.push(item.description || item.question)
            })
          } catch {}
        }
        setSceneHints(hints)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (messages.length > 0) setSceneHintVisible(false)
  }, [messages])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send() {
    const text = input.trim()
    if (!text || streaming) return
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

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const body = await queryStream(text, queryId, controller.signal)
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

  return (
    <div className="page active">
      <div className="chat-container">
        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="chat-message-wrapper">
              <div className="empty-state">
                <h3>{t('chat.welcome.greeting')}</h3>
                <p>{t('chat.welcome.prompt')}</p>
                {sceneHints.length > 0 && sceneHintVisible && (
                  <div className="scene-hint">
                    <div
                      className="scene-hint-header"
                      onClick={() => setSceneHintVisible(false)}
                      style={{ cursor: 'pointer' }}
                    >
                      <span>{t('chat.sceneHint')}</span>
                      <span style={{ fontSize: 12, opacity: 0.6 }}>✕</span>
                    </div>
                    <ul className="scene-hint-list">
                      {sceneHints.map((hint, i) => (
                        <li key={i} className="scene-hint-item">{hint}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {sceneHints.length > 0 && !sceneHintVisible && (
                  <div
                    className="scene-hint-collapsed"
                    onClick={() => setSceneHintVisible(true)}
                    style={{ cursor: 'pointer', marginTop: 12, fontSize: 13, color: 'var(--text-secondary)' }}
                  >
                    {t('knowledge.scene')}: {t('knowledge.sceneEnabled')}
                  </div>
                )}
              </div>
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
