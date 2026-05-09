import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'

const NAV = [
  { to: '/chat',      icon: '💬', label: '对话' },
  { to: '/knowledge', icon: '📚', label: '知识库' },
  { to: '/share',     icon: '🔗', label: '分享中心' },
  { to: '/community', icon: '👥', label: '社区' },
  { to: '/devtools',  icon: '⚡', label: '开发者工具' },
]

export default function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  async function handleLogout() {
    await logout()
    navigate('/login')
  }

  return (
    <div className="flex h-screen bg-slate-900 text-slate-200">
      {/* Sidebar */}
      <aside className="w-14 bg-slate-950 border-r border-slate-800 flex flex-col items-center py-3 gap-1 shrink-0">
        {/* Logo */}
        <div className="w-8 h-8 bg-teal-600 rounded-lg mb-3 flex items-center justify-center text-white text-xs font-bold">
          C
        </div>

        {NAV.map(({ to, icon, label }) => (
          <NavLink
            key={to}
            to={to}
            title={label}
            className={({ isActive }) =>
              `w-10 h-10 rounded-lg flex items-center justify-center text-lg transition-colors ` +
              (isActive ? 'bg-teal-600 text-white' : 'text-slate-400 hover:bg-slate-800 hover:text-white')
            }
          >
            {icon}
          </NavLink>
        ))}

        <div className="flex-1" />

        {/* User avatar / logout */}
        <button
          onClick={handleLogout}
          title={user?.email || '退出'}
          className="w-8 h-8 rounded-full bg-slate-700 hover:bg-slate-600 flex items-center justify-center text-xs text-slate-300 transition-colors"
        >
          {user?.email?.[0]?.toUpperCase() || '?'}
        </button>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  )
}
