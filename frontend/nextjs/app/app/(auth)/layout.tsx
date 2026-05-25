'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { AuthProvider, useAuth } from '@/contexts/AuthContext'
import { ChatProvider } from '@/contexts/ChatContext'
import { I18nProvider } from '@/lib/app-i18n'
import AppLayout from '@/components/app/AppLayout'

function AuthGuard({ children }: { children: React.ReactNode }) {
  const { user } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (user === null) {
      router.replace('/app/login')
    }
  }, [user, router])

  if (user === undefined) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh' }}>
        <div style={{ width: 32, height: 32, border: '3px solid #10A37F', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    )
  }

  if (user === null) {
    return null
  }

  return <AppLayout>{children}</AppLayout>
}

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <I18nProvider>
        <ChatProvider>
          <AuthGuard>{children}</AuthGuard>
        </ChatProvider>
      </I18nProvider>
    </AuthProvider>
  )
}
