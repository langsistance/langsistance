import { Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './contexts/AuthContext'
import Login from './pages/Login'
import Layout from './components/Layout'
import Chat from './pages/Chat'
import Knowledge from './pages/Knowledge'
import Share from './pages/Share'
import Community from './pages/Community'
import DevTools from './pages/DevTools'
import { ChatProvider } from './contexts/ChatContext'

function RequireAuth({ children }) {
  const { user } = useAuth()
  if (user === undefined) return <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', color: 'var(--color-text-secondary)' }}>加载中...</div>
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
            <ChatProvider>
              <Layout />
            </ChatProvider>
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
