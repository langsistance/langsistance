'use client'

import LoginForm from '@/components/app/LoginForm'

interface LoginModalProps {
  actionLabel: string
  onClose: () => void
}

export default function LoginModal({ actionLabel, onClose }: LoginModalProps) {
  function handleOverlayClick(e: React.MouseEvent) {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  return (
    <div className="modal" onClick={handleOverlayClick}>
      <div className="modal-content" style={{ maxWidth: 440 }}>
        <div className="modal-header">
          <h2>{actionLabel}</h2>
          <button className="close-btn" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <div className="modal-body" style={{ padding: '24px 24px 28px' }}>
          {/* LoginForm is self-contained — it handles auth, errors, and i18n */}
          <LoginForm />
        </div>
      </div>
    </div>
  )
}
