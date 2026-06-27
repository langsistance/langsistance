'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { useI18n } from '@/lib/app-i18n'
import LanguageToggleButton from '@/components/app/LanguageToggleButton'
import MessageBell from '@/components/app/MessageBell'
import FeedbackFAB from '@/components/app/FeedbackFAB'
import { getSessions, getLongTaskReportUrl, type SessionItem } from '@/services/api'

const NAV_ITEMS = [
  {
    to: '/app/chat',
    key: 'chat.dialogue',
    icon: (
      <svg className="nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
    ),
  },
  {
    to: '/app/knowledge',
    key: 'knowledge.title',
    icon: (
      <svg className="nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
      </svg>
    ),
  },
  {
    to: '/app/share',
    key: 'share.title',
    icon: (
      <svg className="nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="18" cy="5" r="3" />
        <circle cx="6" cy="12" r="3" />
        <circle cx="18" cy="19" r="3" />
        <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
        <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
      </svg>
    ),
  },
  {
    to: '/app/community',
    key: 'community.title',
    icon: (
      <svg className="nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
      </svg>
    ),
  },
  {
    to: '/app/messages',
    key: 'messages.title',
    icon: (
      <svg className="nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
        <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
      </svg>
    ),
  },
]

const DEVTOOLS_ITEM = {
  to: '/app/devtools',
  key: 'browser.devTools',
  icon: (
    <svg className="nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
    </svg>
  ),
}

function getInitialDevMode() {
  try {
    return localStorage.getItem('devMode') === 'true'
  } catch {
    return false
  }
}

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth()
  const { t } = useI18n()
  const router = useRouter()
  const pathname = usePathname()
  const [devMode, setDevMode] = useState(getInitialDevMode)

  // Derive active session directly from URL on every render
  const [routerKey, setRouterKey] = useState(0)
  useEffect(() => {
    const onNav = () => setRouterKey(k => k + 1)
    window.addEventListener('popstate', onNav)
    return () => window.removeEventListener('popstate', onNav)
  }, [])
  const activeSid = (() => {
    if (typeof window === 'undefined') return null
    return new URLSearchParams(window.location.search).get('session_id')
  })()
  const [menuOpen, setMenuOpen] = useState(false)
  const [sessions, setSessions] = useState<SessionItem[]>([])
  const [sessionsOpen, setSessionsOpen] = useState(true)
  const menuRef = useRef<HTMLDivElement>(null)

  const refreshSessions = useCallback(async () => {
    try {
      const list = await getSessions()
      setSessions(list.slice(0, 20))
    } catch {
      // Non-critical; silently fail
    }
  }, [])

  // Fetch sessions on mount and when route changes (detect long task completion)
  useEffect(() => {
    refreshSessions()
  }, [refreshSessions, pathname])

  // Re-fetch sessions periodically (for long task updates)
  useEffect(() => {
    const timer = setInterval(refreshSessions, 15000)
    return () => clearInterval(timer)
  }, [refreshSessions])

  useEffect(() => {
    if (!menuOpen) return
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [menuOpen])

  async function handleLogout() {
    await logout()
    router.push('/app/login')
  }

  function toggleDevMode() {
    const next = !devMode
    setDevMode(next)
    try { localStorage.setItem('devMode', String(next)) } catch {}
  }

  const visibleNavItems = devMode ? [...NAV_ITEMS, DEVTOOLS_ITEM] : NAV_ITEMS

  return (
    <>
      <header className="header">
        <div className="header-left">
          <div className="logo">
            <div className="logo-icon">
              <img src="/logo.png" alt="Logo" style={{ width: 32, height: 32, borderRadius: 8, objectFit: 'contain' }} />
            </div>
            <span>CopiioAI</span>
          </div>
        </div>
        <div className="header-right">
          <MessageBell />
          <LanguageToggleButton />
          <div className="user-menu" ref={menuRef}>
            <button className="user-avatar" onClick={() => setMenuOpen(v => !v)}>
              {user?.email?.[0]?.toUpperCase() ?? 'U'}
            </button>
            {menuOpen && (
              <div className="user-menu-dropdown">
                <div className="user-menu-email">{user?.email}</div>
                <hr className="user-menu-divider" />
                <button className="user-menu-logout" onClick={handleLogout}>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                    <polyline points="16 17 21 12 16 7" />
                    <line x1="21" y1="12" x2="9" y2="12" />
                  </svg>
                  {t('common.signOut')}
                </button>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="main-container">
        <aside className="sidebar">
          <nav className="nav-menu">
            {visibleNavItems.map(({ to, key, icon }) => (
              <Link key={to} href={to} style={{ textDecoration: 'none' }}>
                <button className={`nav-item${pathname === to ? ' active' : ''}`}>
                  {icon}
                  <span>{t(key)}</span>
                </button>
              </Link>
            ))}
          </nav>

          {sessions.length > 0 && (
            <div className="session-history" data-nav={routerKey}>
              <button
                className={`session-history-header ${sessionsOpen ? 'expanded' : ''}`}
                onClick={() => setSessionsOpen(v => !v)}
              >
                <span className="session-history-title">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                  分析历史
                </span>
                <span className="session-count">{sessions.length}</span>
                <svg
                  className={`session-chevron ${sessionsOpen ? 'open' : ''}`}
                  width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"
                >
                  <polyline points="6 9 12 15 18 9" />
                </svg>
              </button>
              {sessionsOpen && (
                <div className="session-list">
                  {sessions.map((s) => (
                    <a
                      key={s.session_id}
                      href={`/app/chat?session_id=${s.session_id}`}
                      className={`session-item${pathname === '/app/chat' && activeSid === s.session_id ? ' session-active' : ''}`}
                      title={s.title}
                      style={{ textDecoration: 'none', color: 'inherit', display: 'block' }}
                    >
                        <div className="session-item-main">
                          <span className="session-item-title">{s.title || '专利分析'}</span>
                          <span className="session-item-time">
                            {s.update_time
                              ? new Date(s.update_time).toLocaleDateString('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
                              : ''}
                          </span>
                        </div>
                        {s.long_task_ids && s.long_task_ids.length > 0 && (
                          <div className="session-item-actions" onClick={e => e.preventDefault()}>
                            {s.long_task_ids.map((tid) => (
                              <a
                                key={tid}
                                href={getLongTaskReportUrl(tid, 'docx')}
                                className="session-report-link"
                                target="_blank"
                                rel="noopener noreferrer"
                                title={`下载报告 ${tid}`}
                              >
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                  <polyline points="7 10 12 15 17 10" />
                                  <line x1="12" y1="15" x2="12" y2="3" />
                                </svg>
                                DOCX
                              </a>
                            ))}
                          </div>
                        )}
                      </a>
                  ))}
                </div>
              )}
            </div>
          )}

          <div className="sidebar-footer">
            <button className="nav-item" onClick={toggleDevMode} style={{ cursor: 'pointer' }}>
              <span>{t('developer.pattern')}</span>
              <div className="switch-wrap" style={{ marginLeft: 'auto' }}>
                <div className={`switch-container${devMode ? ' active' : ''}`}>
                  <div className="switch-slider" />
                </div>
              </div>
            </button>
          </div>
        </aside>

        <main className="content-area">
          {children}
        </main>
      </div>
      <FeedbackFAB />
    </>
  )
}
