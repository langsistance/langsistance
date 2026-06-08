'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { getUnreadCount } from '@/services/api'
import { useI18n } from '@/lib/app-i18n'

export default function MessageBell() {
  const { t } = useI18n()
  const router = useRouter()
  const [unreadCount, setUnreadCount] = useState(0)

  const fetchUnreadCount = useCallback(async () => {
    try {
      const result = await getUnreadCount()
      if (result?.success) {
        setUnreadCount(result.unread_count || 0)
      }
    } catch {
      // Silently fail for polling
    }
  }, [])

  useEffect(() => {
    fetchUnreadCount()
    const timer = setInterval(fetchUnreadCount, 30000)
    return () => clearInterval(timer)
  }, [fetchUnreadCount])

  function handleClick() {
    router.push('/app/messages')
  }

  return (
    <button className="message-bell-btn" onClick={handleClick} title={t('messages.bellTitle')}>
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
        <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
      </svg>
      {unreadCount > 0 && (
        <span className="unread-badge">{unreadCount > 99 ? '99+' : unreadCount}</span>
      )}
    </button>
  )
}
