'use client'

import { useState, useEffect, useCallback } from 'react'
import { getMessages, markMessageRead, markAllMessagesRead } from '@/services/api'
import { useI18n } from '@/lib/app-i18n'

interface Message {
  id: number
  title: string
  content: string
  is_read: boolean
  create_time: string
  feedback_id: number | null
}

export default function MessagesPage() {
  const { t } = useI18n()
  const [messages, setMessages] = useState<Message[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedId, setExpandedId] = useState<number | null>(null)

  const loadMessages = useCallback(async () => {
    try {
      const result = await getMessages()
      if (result?.success) {
        setMessages(result.data || [])
      }
    } catch (err) {
      console.error('Load messages error:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadMessages()
  }, [loadMessages])

  async function handleMarkRead(messageId: number) {
    // Optimistic update
    setMessages(prev => prev.map(m =>
      m.id === messageId ? { ...m, is_read: true } : m
    ))
    try {
      await markMessageRead(messageId)
    } catch {
      // Revert on failure
      setMessages(prev => prev.map(m =>
        m.id === messageId ? { ...m, is_read: false } : m
      ))
    }
  }

  async function handleMarkAllRead() {
    const hadUnread = messages.some(m => !m.is_read)
    if (!hadUnread) return
    // Optimistic update
    setMessages(prev => prev.map(m => ({ ...m, is_read: true })))
    try {
      await markAllMessagesRead()
    } catch {
      setMessages(prev => prev.map(m => ({ ...m, is_read: false })))
    }
  }

  function toggleExpand(messageId: number) {
    setExpandedId(prev => prev === messageId ? null : messageId)
  }

  function escapeHtml(text: string): string {
    const div = document.createElement('div')
    div.textContent = text
    return div.innerHTML
  }

  const hasUnread = messages.some(m => !m.is_read)

  return (
    <div className="page active">
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>消息通知</h1>
            <p>查看来自 CopiioAI 团队的消息和回复</p>
          </div>
          {hasUnread && (
            <button className="btn btn-secondary btn-sm" onClick={handleMarkAllRead}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="20 6 9 17 4 12"/>
              </svg>
              全部已读
            </button>
          )}
        </div>
      </div>
      <div className="page-content">
        {loading ? (
          <div className="empty-state"><p>加载中...</p></div>
        ) : messages.length === 0 ? (
          <div className="empty-state">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ opacity: 0.3 }}>
              <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
              <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
            </svg>
            <p>暂无消息</p>
            <p style={{ fontSize: 13, color: 'var(--color-text-secondary)' }}>当有新的回复时，会在这里显示</p>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map(msg => {
              const isUnread = !msg.is_read
              const isExpanded = expandedId === msg.id
              return (
                <div
                  key={msg.id}
                  className={`message-card${isUnread ? ' unread' : ''}${isExpanded ? ' message-expanded' : ''}`}
                  onClick={() => {
                    if (isUnread) handleMarkRead(msg.id)
                    toggleExpand(msg.id)
                  }}
                >
                  <div className="message-card-header">
                    <h3 className="message-card-title">{msg.title}</h3>
                    <span className="message-card-time">{msg.create_time}</span>
                  </div>
                  <div
                    className="message-card-body"
                    dangerouslySetInnerHTML={{ __html: escapeHtml(msg.content).replace(/\n/g, '<br/>') }}
                  />
                  {isUnread && (
                    <div className="message-card-footer">
                      <span className="unread-dot" />
                      <span>未读</span>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
