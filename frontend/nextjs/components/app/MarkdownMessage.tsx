'use client'

import { useMemo, useState } from 'react'
import { marked } from 'marked'
import { markedHighlight } from 'marked-highlight'
import hljs from 'highlight.js'
import 'highlight.js/styles/github.css'
import { useI18n } from '@/lib/app-i18n'

marked.use({ gfm: true, breaks: true })
marked.use(
  markedHighlight({
    langPrefix: 'hljs language-',
    highlight(code, lang) {
      const language = hljs.getLanguage(lang) ? lang : 'plaintext'
      return hljs.highlight(code, { language }).value
    },
  })
)

interface Props {
  content: string
  streaming: boolean
}

export default function MarkdownMessage({ content, streaming }: Props) {
  const { t } = useI18n()
  const [copied, setCopied] = useState(false)
  const [downloaded, setDownloaded] = useState(false)

  const html = useMemo(() => {
    if (!content || streaming) return ''
    return marked.parse(content) as string
  }, [content, streaming])

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

  if (streaming) {
    return (
      <div className="chat-message assistant">
        {content || '▋'}
      </div>
    )
  }

  return (
    <div className="chat-message assistant">
      <div dangerouslySetInnerHTML={{ __html: html }} />
      {content.trim() && (
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
