'use client'
import { createContext, useContext, useCallback, useEffect, useState, useRef } from 'react'
import { onAuthChange, logout as clientLogout, type AuthUser } from '@/lib/auth-client'
import LoginModal from '@/components/app/LoginModal'

interface AuthContextValue {
  user: AuthUser | null | undefined
  logout: () => Promise<void>
  requireAuth: (action: () => void, label?: string) => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null | undefined>(undefined)
  const [loginModalOpen, setLoginModalOpen] = useState(false)
  const [actionLabel, setActionLabel] = useState('')
  const pendingActionRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    return onAuthChange(setUser)
  }, [])

  // When user becomes logged in, execute pending action
  useEffect(() => {
    if (user && pendingActionRef.current) {
      const action = pendingActionRef.current
      pendingActionRef.current = null
      setLoginModalOpen(false)
      // Execute after state updates settle
      setTimeout(() => action(), 0)
    }
  }, [user])

  const logout = useCallback(async () => {
    clientLogout()
  }, [])

  function requireAuth(action: () => void, label?: string) {
    if (user) {
      action()
      return
    }
    pendingActionRef.current = action
    setActionLabel(label || 'Sign in to continue')
    setLoginModalOpen(true)
  }

  function closeLogin() {
    pendingActionRef.current = null
    setLoginModalOpen(false)
  }

  return (
    <AuthContext.Provider value={{ user, logout, requireAuth }}>
      {children}
      {loginModalOpen && (
        <LoginModal
          actionLabel={actionLabel}
          onClose={closeLogin}
        />
      )}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
