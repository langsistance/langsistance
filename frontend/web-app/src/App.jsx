import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './contexts/AuthContext'
import Login from './pages/Login'
import Layout from './components/Layout'
import Chat from './pages/Chat'
import Knowledge from './pages/Knowledge'
import Share from './pages/Share'
import Community from './pages/Community'
import DevTools from './pages/DevTools'

function RequireAuth({ children }) {
  const { user } = useAuth()
  if (user === undefined) return <div className="flex items-center justify-center h-screen text-slate-400">加载中...</div>
  if (!user) return <Navigate to="/login" replace />
  return children
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        path="/*"
        element={
          <RequireAuth>
            <Layout />
          </RequireAuth>
        }
      >
        <Route index element={<Navigate to="/chat" replace />} />
        <Route path="chat" element={<Chat />} />
        <Route path="knowledge" element={<Knowledge />} />
        <Route path="share" element={<Share />} />
        <Route path="community" element={<Community />} />
        <Route path="devtools" element={<DevTools />} />
      </Route>
    </Routes>
  )
}
