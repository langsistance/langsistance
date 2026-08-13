import { Routes, Route } from 'react-router-dom'
import Login from './pages/Login'
import Layout from './components/Layout'
import Chat from './pages/Chat'
import Knowledge from './pages/Knowledge'
import Share from './pages/Share'
import Community from './pages/Community'
import DevTools from './pages/DevTools'
import { ChatProvider } from './contexts/ChatContext'

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route
        path="/*"
        element={
          <ChatProvider>
            <Layout />
          </ChatProvider>
        }
      >
        <Route index element={<Chat />} />
        <Route path="chat" element={<Chat />} />
        <Route path="knowledge" element={<Knowledge />} />
        <Route path="share" element={<Share />} />
        <Route path="community" element={<Community />} />
        <Route path="devtools" element={<DevTools />} />
      </Route>
    </Routes>
  )
}
