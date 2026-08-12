import { createContext, useContext, useEffect, useState, useRef } from 'react'
import { onAuthStateChanged, signOut } from 'firebase/auth'
import { auth } from '../firebase'
import LoginModal from '../components/LoginModal'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(undefined) // undefined = loading, null = anonymous, object = logged in
  const [loginModalOpen, setLoginModalOpen] = useState(false)
  const [actionLabel, setActionLabel] = useState('')
  const pendingActionRef = useRef(null)

  useEffect(() => {
    return onAuthStateChanged(auth, setUser)
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

  const logout = () => signOut(auth)

  function requireAuth(action, label) {
    if (user) {
      // Already logged in, execute immediately
      action()
      return
    }
    // Store action and open login modal
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

export const useAuth = () => useContext(AuthContext)
