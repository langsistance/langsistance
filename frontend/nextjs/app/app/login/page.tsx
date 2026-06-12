'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { AuthProvider, useAuth } from '@/contexts/AuthContext'
import { I18nProvider } from '@/lib/app-i18n'
import LoginForm from '@/components/app/LoginForm'

function LoginPageContent() {
  const { user } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (user) {
      router.replace('/')
    }
  }, [user, router])

  if (user) {
    return null
  }

  return <LoginForm />
}

export default function LoginPage() {
  return (
    <AuthProvider>
      <I18nProvider>
        <LoginPageContent />
      </I18nProvider>
    </AuthProvider>
  )
}
