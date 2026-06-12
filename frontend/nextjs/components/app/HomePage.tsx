'use client'

import '@/styles/app.css'
import { AuthProvider, useAuth } from '@/contexts/AuthContext'
import { ChatProvider } from '@/contexts/ChatContext'
import { I18nProvider } from '@/lib/app-i18n'
import AppLayout from '@/components/app/AppLayout'
import SceneOnboardingModal from '@/components/app/SceneOnboardingModal'
import LoginForm from '@/components/app/LoginForm'
import Chat from '@/app/app/(auth)/chat/page'

function HomePageContent() {
  const { user } = useAuth()

  if (user === undefined) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}>
        <div style={{ width: 32, height: 32, border: '3px solid #10A37F', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    )
  }

  if (user === null) {
    return <LoginForm />
  }

  return (
    <AppLayout>
      <SceneOnboardingModal />
      <Chat />
    </AppLayout>
  )
}

export default function HomePage() {
  return (
    <AuthProvider>
      <I18nProvider>
        <ChatProvider>
          <HomePageContent />
        </ChatProvider>
      </I18nProvider>
    </AuthProvider>
  )
}
