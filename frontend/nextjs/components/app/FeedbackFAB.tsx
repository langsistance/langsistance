'use client'

import { useState } from 'react'
import FeedbackModal from './FeedbackModal'

export default function FeedbackFAB() {
  const [modalOpen, setModalOpen] = useState(false)

  return (
    <>
      <button
        className="feedback-fab"
        onClick={() => setModalOpen(true)}
        title="用户反馈"
      >
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </button>
      <FeedbackModal open={modalOpen} onClose={() => setModalOpen(false)} />
    </>
  )
}
