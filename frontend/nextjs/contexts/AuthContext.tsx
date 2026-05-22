'use client'
import { createContext, useContext, useCallback, useEffect, useState } from 'react'
import { onAuthChange, logout as clientLogout, type AuthUser } from '@/lib/auth-client'

interface AuthContextValue {
  user: AuthUser | null | undefined
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null | undefined>(undefined)

  useEffect(() => {
    return onAuthChange(setUser)
  }, [])

  const logout = useCallback(async () => {
    clientLogout()
  }, [])

  return (
    <AuthContext.Provider value={{ user, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
