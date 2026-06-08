'use client'

import { useState, useRef, useEffect } from 'react'
import { submitFeedback } from '@/services/api'

interface FeedbackModalProps {
  open: boolean
  onClose: () => void
}

export default function FeedbackModal({ open, onClose }: FeedbackModalProps) {
  const [content, setContent] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (open) {
      setContent('')
      setStatus('idle')
      setTimeout(() => textareaRef.current?.focus(), 100)
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    function handleEsc(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleEsc)
    return () => document.removeEventListener('keydown', handleEsc)
  }, [open, onClose])

  async function handleSubmit() {
    const trimmed = content.trim()
    if (!trimmed || submitting) return

    setSubmitting(true)
    setStatus('idle')
    try {
      const result = await submitFeedback(trimmed)
      if (result?.success) {
        setStatus('success')
        setTimeout(onClose, 1500)
      } else {
        setStatus('error')
      }
    } catch {
      setStatus('error')
    } finally {
      setSubmitting(false)
    }
  }

  if (!open) return null

  return (
    <div className="feedback-modal">
      <div className="feedback-modal-overlay" onClick={onClose} />
      <div className="feedback-modal-content">
        <div className="feedback-modal-header">
          <h2>💬 用户反馈</h2>
          <button className="feedback-modal-close" onClick={onClose}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"/>
              <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
        <div className="feedback-modal-body">
          <div className="feedback-thanks">
            <p>🙏 感谢您使用 CopiioAI！</p>
            <p>我们认真聆听每一位用户的反馈，您的意见对我们至关重要。</p>
            <p>请告诉我们您的想法、建议或遇到的问题：</p>
          </div>
          <textarea
            ref={textareaRef}
            className="feedback-textarea"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="请输入您的反馈..."
            rows={5}
            maxLength={2000}
          />
          <div className="feedback-modal-footer">
            <span className="feedback-char-count">{content.length}/2000</span>
            <div className="feedback-modal-actions">
              <button className="btn btn-secondary" onClick={onClose}>取消</button>
              <button
                className="btn btn-primary"
                onClick={handleSubmit}
                disabled={!content.trim() || submitting}
              >
                {submitting ? '提交中...' : status === 'success' ? '已提交 ✓' : '提交反馈'}
              </button>
            </div>
          </div>
          {status === 'error' && (
            <p style={{ color: '#EF4444', fontSize: 13, marginTop: 8 }}>提交失败，请稍后重试</p>
          )}
        </div>
      </div>
    </div>
  )
}
