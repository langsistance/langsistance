'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import 'highlight.js/styles/github.css'
import { useI18n } from '@/lib/app-i18n'
import { attachImageRetryHandlers } from '@/lib/imageRetry'
import {
  shouldShowAssistantTransientStatus,
  shouldShowAssistantWaiting,
} from '@/lib/messagePresentation'
import { renderMarkdownToHtml } from '@/lib/markdownRender'

interface Props {
  content: string
  streaming: boolean
  transientStatus?: string
}

const THROTTLE_MS = 1000

export default function MarkdownMessage({ content, streaming, transientStatus = '' }: Props) {
  const { t } = useI18n()
  const [copied, setCopied] = useState(false)
  const [downloaded, setDownloaded] = useState(false)
  const [html, setHtml] = useState('')

  const lastRenderTimeRef = useRef(0)
  const pendingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const latestContentRef = useRef(content)
  const latestStreamingRef = useRef(streaming)
  const messageContentRef = useRef<HTMLDivElement | null>(null)
  const showWaiting = shouldShowAssistantWaiting(content, streaming)
  const showTransientStatus = shouldShowAssistantTransientStatus(transientStatus, streaming)

  const doRender = useCallback((text: string, isStreaming: boolean) => {
    const src = isStreaming ? text + ' ▋' : text
    setHtml(renderMarkdownToHtml(src) as string)
    lastRenderTimeRef.current = Date.now()
  }, [])

  useEffect(() => {
    latestContentRef.current = content
    latestStreamingRef.current = streaming

    if (!content) {
      if (pendingTimerRef.current) {
        clearTimeout(pendingTimerRef.current)
        pendingTimerRef.current = null
      }
      setHtml('')
      return
    }

    if (!streaming) {
      // Streaming done: flush immediately, cancel any pending throttle
      if (pendingTimerRef.current) {
        clearTimeout(pendingTimerRef.current)
        pendingTimerRef.current = null
      }
      doRender(content, false)
      return
    }

    // Streaming: throttle re-parses to at most once per THROTTLE_MS
    const elapsed = Date.now() - lastRenderTimeRef.current
    if (pendingTimerRef.current) {
      clearTimeout(pendingTimerRef.current)
      pendingTimerRef.current = null
    }

    if (elapsed >= THROTTLE_MS) {
      doRender(content, true)
    } else {
      pendingTimerRef.current = setTimeout(() => {
        pendingTimerRef.current = null
        doRender(latestContentRef.current, latestStreamingRef.current)
      }, THROTTLE_MS - elapsed)
    }
  }, [content, streaming, doRender])

  useEffect(() => {
    return () => {
      if (pendingTimerRef.current) clearTimeout(pendingTimerRef.current)
    }
  }, [])

  useEffect(() => {
    if (!messageContentRef.current || !html) return

    return attachImageRetryHandlers(messageContentRef.current)
  }, [html])

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {}
  }

  function handleDownload() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5)
    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `CopiioAI_Chat_${timestamp}.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    setDownloaded(true)
    setTimeout(() => setDownloaded(false), 2000)
  }

  return (
    <div ref={messageContentRef} className={`chat-message assistant${showWaiting ? ' assistant-is-waiting' : ''}`}>
      {showWaiting && (
        <div className="assistant-waiting" role="status" aria-live="polite" aria-label={t('chat.processing')}>
          <span className="assistant-waiting-orbit" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span className="assistant-waiting-copy">
            <span className="assistant-waiting-title">{t('chat.processing')}</span>
            {transientStatus && (
              <span className="assistant-waiting-detail">{transientStatus}</span>
            )}
          </span>
          <span className="assistant-waiting-scan" aria-hidden="true" />
        </div>
      )}
      <div dangerouslySetInnerHTML={{ __html: html || '▋' }} />
      {!showWaiting && showTransientStatus && (
        <div className="assistant-transient-status" role="status" aria-live="polite">
          <span className="assistant-transient-status-dot" aria-hidden="true" />
          <span>{transientStatus}</span>
        </div>
      )}
      {!streaming && content.trim() && (
        <div className="message-action-buttons">
          <button
            className={`copy-button${copied ? ' copied' : ''}`}
            onClick={handleCopy}
            title={t('chat.copy')}
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
          <button
            className={`download-button${downloaded ? ' downloaded' : ''}`}
            onClick={handleDownload}
            title={t('chat.download')}
          >
            {downloaded ? (
              <svg viewBox="0 0 24 24" fill="none" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            )}
          </button>
        </div>
      )}
    </div>
  )
}
