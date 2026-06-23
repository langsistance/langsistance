import { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { listSessions } from '../services/sessionService'

/**
 * SessionSidebar — displays a list of user sessions and allows selection.
 *
 * Props:
 *   activeSessionId  — currently selected session ID (or null)
 *   onSelectSession  — callback(sessionId) when user clicks a session
 *   onDeleteSession  — callback(sessionId) when user deletes/archives a session
 */
export default function SessionSidebar({ activeSessionId, onSelectSession, onDeleteSession }) {
  const { user } = useAuth()
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(false)
  const [collapsed, setCollapsed] = useState(false)

  const loadSessions = useCallback(async () => {
    if (!user) return
    setLoading(true)
    try {
      const res = await listSessions(user.uid)
      const list = res?.sessions || res?.data?.sessions || []
      setSessions(list)
    } catch (err) {
      console.error('Failed to load sessions:', err)
    } finally {
      setLoading(false)
    }
  }, [user])

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  // Refresh when activeSessionId changes (e.g. a new session was created)
  useEffect(() => {
    if (activeSessionId) {
      loadSessions()
    }
  }, [activeSessionId, loadSessions])

  function handleDelete(e, sessionId) {
    e.stopPropagation()
    if (onDeleteSession) {
      onDeleteSession(sessionId)
      // Optimistic removal
      setSessions((prev) => prev.filter((s) => s.session_id !== sessionId))
    }
  }

  function formatTime(timeStr) {
    if (!timeStr) return ''
    try {
      const d = new Date(timeStr)
      const now = new Date()
      const diff = now - d
      const days = Math.floor(diff / 86400000)
      if (days === 0) {
        const hours = Math.floor(diff / 3600000)
        if (hours === 0) {
          const mins = Math.floor(diff / 60000)
          return `${mins}分钟前`
        }
        return `${hours}小时前`
      }
      if (days < 7) return `${days}天前`
      return d.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
    } catch {
      return timeStr
    }
  }

  function sessionLabel(s) {
    return s.title || `分析任务 ${s.session_id?.slice(0, 10) || ''}`
  }

  return (
    <div style={{
      width: collapsed ? 44 : 220,
      flexShrink: 0,
      borderRight: '1px solid var(--color-border)',
      background: 'var(--color-bg-white)',
      display: 'flex',
      flexDirection: 'column',
      transition: 'width 0.2s ease',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: collapsed ? '12px 10px' : '12px 14px',
        borderBottom: '1px solid var(--color-border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        minHeight: 44,
      }}>
        {!collapsed && (
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text-primary)' }}>
            对话历史
          </span>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            color: 'var(--color-text-secondary)',
            padding: 4,
            display: 'flex',
            borderRadius: 4,
            transition: 'transform 0.2s',
            transform: collapsed ? 'rotate(180deg)' : 'none',
          }}
          title={collapsed ? '展开侧栏' : '收起侧栏'}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
        </button>
      </div>

      {/* Session list */}
      {!collapsed && (
        <div style={{ flex: 1, overflowY: 'auto', padding: '4px 0' }}>
          {loading && sessions.length === 0 && (
            <div style={{ padding: '16px 14px', textAlign: 'center' }}>
              <div className="loading" style={{ margin: '0 auto' }} />
            </div>
          )}

          {!loading && sessions.length === 0 && (
            <div style={{
              padding: '20px 14px',
              textAlign: 'center',
              color: 'var(--color-text-secondary)',
              fontSize: 13,
            }}>
              暂无对话记录
            </div>
          )}

          {sessions.map((s) => {
            const isActive = s.session_id === activeSessionId
            return (
              <div
                key={s.session_id}
                onClick={() => onSelectSession?.(s.session_id)}
                style={{
                  padding: '10px 14px',
                  cursor: 'pointer',
                  background: isActive ? '#F0FDF4' : 'transparent',
                  borderLeft: isActive ? '3px solid #10A37F' : '3px solid transparent',
                  transition: 'background 0.15s',
                }}
                onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = 'var(--color-bg-hover)' }}
                onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = 'transparent' }}
              >
                <div style={{ fontSize: 13, fontWeight: isActive ? 600 : 400, color: '#111827', marginBottom: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {sessionLabel(s)}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: 11, color: '#9CA3AF' }}>
                    {formatTime(s.update_time || s.create_time)}
                  </span>
                  {onDeleteSession && (
                    <button
                      onClick={(e) => handleDelete(e, s.session_id)}
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        color: '#9CA3AF',
                        padding: 2,
                        display: 'flex',
                        opacity: 0,
                        borderRadius: 3,
                      }}
                      onMouseEnter={(e) => { e.currentTarget.style.opacity = '1' }}
                      onMouseLeave={(e) => { e.currentTarget.style.opacity = '0' }}
                      title="删除"
                    >
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="3 6 5 6 21 6" />
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
