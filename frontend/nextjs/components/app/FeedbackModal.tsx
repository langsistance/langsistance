'use client'

import { useState, useRef, useEffect } from 'react'
import { submitFeedback } from '@/services/api'
import { useI18n } from '@/lib/app-i18n'

interface FeedbackModalProps {
  open: boolean
  onClose: () => void
}

export default function FeedbackModal({ open, onClose }: FeedbackModalProps) {
  const { t } = useI18n()
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
        showToast(t('feedback.submitSuccess'))
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

  function showToast(message: string) {
    const toast = document.createElement('div')
    toast.className = 'global-toast success show'
    toast.textContent = message
    document.body.appendChild(toast)
    setTimeout(() => {
      toast.classList.remove('show')
      setTimeout(() => toast.remove(), 300)
    }, 2500)
  }

  if (!open) return null

  return (
    <div className="feedback-modal">
      <div className="feedback-modal-overlay" onClick={onClose} />
      <div className="feedback-modal-content">
        <div className="feedback-modal-header">
          <h2>💬 {t('feedback.title')}</h2>
          <button className="feedback-modal-close" onClick={onClose}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"/>
              <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>
        <div className="feedback-modal-body">
          <div className="feedback-thanks">
            <p>{t('feedback.thanksTitle')}</p>
            <p>{t('feedback.thanksDesc')}</p>
            <p>{t('feedback.thanksPrompt')}</p>
          </div>
          <textarea
            ref={textareaRef}
            className="feedback-textarea"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder={t('feedback.placeholder')}
            rows={5}
            maxLength={2000}
          />
          <div className="feedback-modal-footer">
            <span className="feedback-char-count">{content.length}/2000</span>
            <div className="feedback-modal-actions">
              <button className="btn btn-secondary" onClick={onClose}>{t('common.cancel')}</button>
              <button
                className="btn btn-primary"
                onClick={handleSubmit}
                disabled={!content.trim() || submitting}
              >
                {submitting
                  ? t('feedback.submitting')
                  : status === 'success'
                    ? t('feedback.submitted')
                    : t('feedback.submit')}
              </button>
            </div>
          </div>
          {status === 'error' && (
            <p style={{ color: '#EF4444', fontSize: 13, marginTop: 8 }}>{t('feedback.submitFailed')}</p>
          )}
        </div>
      </div>
    </div>
  )
}
