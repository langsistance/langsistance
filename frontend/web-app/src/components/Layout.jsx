import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

const NAV = [
  { to: '/chat',      label: '对话' },
  { to: '/knowledge', label: '知识库' },
  { to: '/share',     label: '分享中心' },
  { to: '/community', label: '社区' },
  { to: '/devtools',  label: '开发者工具' },
]

export default function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  async function handleLogout() {
    await logout()
    navigate('/login')
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* Sidebar nav */}
      <nav className="nav-menu" style={{ width: 180, flexShrink: 0, borderRight: '1px solid var(--color-border)', display: 'flex', flexDirection: 'column', background: 'var(--color-bg-white)' }}>
        <div style={{ padding: '16px 12px 8px', borderBottom: '1px solid var(--color-border)' }}>
          <span style={{ fontWeight: 700, fontSize: 15, color: 'var(--color-text-primary)' }}>CopiioAI</span>
        </div>
        <div style={{ flex: 1, padding: '8px 0' }}>
          {NAV.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
            >
              {label}
            </NavLink>
          ))}
        </div>
        <div style={{ padding: '8px 12px', borderTop: '1px solid var(--color-border)' }}>
          <button
            onClick={handleLogout}
            className="btn btn-secondary"
            style={{ width: '100%', fontSize: 13 }}
          >
            {user?.email?.split('@')[0] || '退出'} · 退出
          </button>
        </div>
      </nav>

      {/* Page content */}
      <div className="content-area" style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Outlet />
      </div>
    </div>
  )
}
