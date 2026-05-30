'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { usePathname, useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { useI18n } from '@/lib/app-i18n'
import LanguageToggleButton from '@/components/app/LanguageToggleButton'

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
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

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
    </>
  )
}
