import { useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useI18n } from '../i18n'

const NAV_ITEMS = [
  {
    to: '/chat',
    key: 'chat.dialogue',
    icon: (
      <svg className="nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
    ),
  },
  {
    to: '/knowledge',
    key: 'knowledge.title',
    icon: (
      <svg className="nav-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
      </svg>
    ),
  },
  {
    to: '/share',
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
    to: '/community',
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
  to: '/devtools',
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

export default function Layout() {
  const { user, logout } = useAuth()
  const { lang, setLang, t } = useI18n()
  const navigate = useNavigate()
  const [devMode, setDevMode] = useState(getInitialDevMode)

  async function handleLogout() {
    await logout()
    navigate('/login')
  }

  function toggleLang() {
    setLang(lang === 'en' ? 'zh' : 'en')
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
              <img src="/app/logo.png" alt="Logo" style={{ width: 32, height: 32, borderRadius: 8, objectFit: 'contain' }} />
            </div>
            <span>CopiioAI</span>
          </div>
        </div>
        <div className="header-right">
          <button className="language-toggle-btn" onClick={toggleLang} title={lang === 'en' ? 'Switch to 中文' : 'Switch to English'}>
            <span>{lang === 'en' ? '🇺🇸' : '🇨🇳'}</span>
            <span>{lang === 'en' ? 'English' : '中文'}</span>
          </button>
          <button className="user-avatar-btn" onClick={handleLogout} title={`${t('common.back')} (${user?.email})`}>
            {user?.email?.[0]?.toUpperCase() || 'U'}
          </button>
        </div>
      </header>

      <div className="main-container">
        <aside className="sidebar">
          <nav className="nav-menu">
            {visibleNavItems.map(({ to, key, icon }) => (
              <NavLink key={to} to={to} style={{ textDecoration: 'none' }}>
                {({ isActive }) => (
                  <button className={`nav-item${isActive ? ' active' : ''}`}>
                    {icon}
                    <span>{t(key)}</span>
                  </button>
                )}
              </NavLink>
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
          <Outlet />
        </main>
      </div>
    </>
  )
}
