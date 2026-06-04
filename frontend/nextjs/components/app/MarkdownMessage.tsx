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
import { copyTextToClipboard } from '@/lib/clipboard'
import { artifactVisualLabel, orderDownloadArtifacts } from '@/lib/downloadArtifacts'

interface Props {
  content: string
  artifacts?: ChatArtifact[]
  streaming: boolean
  transientStatus?: string
}

interface ChatArtifact {
  artifactId: string
  format: string
  filename: string
  mimeType: string
  rowCount: number
  columnCount: number
  chunks: string[]
  complete: boolean
}

const THROTTLE_MS = 1000

function artifactLabel(artifact: ChatArtifact, t: (key: string) => string) {
  return artifact.format === 'csv' ? t('chat.downloadCsv') : t('chat.downloadExcel')
}

function artifactIcon(format: string) {
  const label = artifactVisualLabel(format)
  if (format === 'xlsx') {
    return (
      <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z" />
        <polyline points="14 2 14 7 19 7" />
        <rect className="artifact-icon-fill" x="7.2" y="8.9" width="9.6" height="6.5" rx="0.8" />
        <path d="M7.2 11.1h9.6M10.4 8.9v6.5M13.6 8.9v6.5" />
        <text className="artifact-icon-label" x="12" y="19.7" textAnchor="middle" fontSize="4.4" fontWeight="800">
          {label}
        </text>
      </svg>
    )
  }

  return (
    <svg viewBox="0 0 24 24" fill="none" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z" />
      <polyline points="14 2 14 7 19 7" />
      <path d="M7.4 9.4h5.8M7.4 12.2h7.8M7.4 15h3.2" />
      <text className="artifact-icon-label artifact-icon-comma" x="13.2" y="16.2" fontSize="5.8" fontWeight="800">,</text>
      <text className="artifact-icon-label artifact-icon-comma" x="15.4" y="16.2" fontSize="5.8" fontWeight="800">,</text>
      <text className="artifact-icon-label" x="12" y="20" textAnchor="middle" fontSize="5" fontWeight="800">
        {label}
      </text>
    </svg>
  )
}

function base64ChunksToBlob(chunks: string[], mimeType: string) {
  const byteArrays = chunks.map((chunk) => {
    const binary = window.atob(chunk)
    const bytes = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i)
    }
    return bytes
  })
  return new Blob(byteArrays, { type: mimeType })
}

export default function MarkdownMessage({ content, artifacts = [], streaming, transientStatus = '' }: Props) {
  const { t } = useI18n()
  const [copied, setCopied] = useState(false)
  const [downloaded, setDownloaded] = useState(false)
  const [downloadedArtifactId, setDownloadedArtifactId] = useState<string | null>(null)
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
    if (await copyTextToClipboard(content)) {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
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

  function handleArtifactDownload(artifact: ChatArtifact) {
    const blob = base64ChunksToBlob(artifact.chunks || [], artifact.mimeType)
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = artifact.filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    setDownloadedArtifactId(artifact.artifactId)
    setTimeout(() => setDownloadedArtifactId(null), 2000)
  }

  const orderedArtifacts = orderDownloadArtifacts(artifacts) as ChatArtifact[]
  const showActions = !streaming && Boolean(content.trim() || orderedArtifacts.length)

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
      {showActions && (
        <div className="message-action-buttons">
          {content.trim() && (
            <button
              className={`copy-button${copied ? ' copied' : ''}`}
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
          )}
          {orderedArtifacts.map((artifact) => (
            <button
              key={artifact.artifactId}
              className={`download-button artifact-download-button ${artifact.format}${downloadedArtifactId === artifact.artifactId ? ' downloaded' : ''}`}
              onClick={() => handleArtifactDownload(artifact)}
              data-tooltip={artifactLabel(artifact, t)}
              data-downloaded-tooltip={t('chat.downloaded')}
              aria-label={artifactLabel(artifact, t)}
            >
              {downloadedArtifactId === artifact.artifactId ? (
                <svg viewBox="0 0 24 24" fill="none" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              ) : (
                artifactIcon(artifact.format)
              )}
            </button>
          ))}
          {content.trim() && (
            <button
              className={`download-button${downloaded ? ' downloaded' : ''}`}
              onClick={handleDownload}
              data-tooltip={t('chat.download')}
              data-downloaded-tooltip={t('chat.downloaded')}
              aria-label={t('chat.downloadContent')}
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
          )}
        </div>
      )}
    </div>
  )
}
