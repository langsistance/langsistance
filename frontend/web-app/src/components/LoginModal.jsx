import { useState } from 'react'
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
} from 'firebase/auth'
import { auth } from '../firebase'
import { useI18n } from '../i18n'

export default function LoginModal({ actionLabel, onClose }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isSignUp, setIsSignUp] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const { t, lang } = useI18n()

  async function handleEmailAuth(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      if (isSignUp) {
        await createUserWithEmailAndPassword(auth, email, password)
      } else {
        await signInWithEmailAndPassword(auth, email, password)
      }
      // onAuthStateChanged will fire, AuthContext will handle pending action
    } catch (err) {
      setError(err.message)
      setLoading(false)
    }
  }

  async function handleGoogle() {
    setError('')
    setLoading(true)
    try {
      await signInWithPopup(auth, new GoogleAuthProvider())
      // onAuthStateChanged fires, AuthContext handles pending action
    } catch (err) {
      if (err.code !== 'auth/popup-closed-by-user') {
        setError(err.message)
      }
      setLoading(false)
    }
  }

  function handleOverlayClick(e) {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  return (
    <div className="modal" onClick={handleOverlayClick}>
      <div className="modal-overlay" />
      <div className="modal-content" style={{ maxWidth: 420 }}>
        <div className="modal-header">
          <h2>{actionLabel || (isSignUp ? (lang === 'en' ? 'Create Account' : '注册') : (lang === 'en' ? 'Sign In' : '登录'))}</h2>
          <button className="modal-close-btn" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="modal-body">
          {error && (
            <div style={{ padding: '10px 14px', background: '#FFF0F0', color: '#D32F2F', borderRadius: 8, fontSize: 13, marginBottom: 16 }}>
              {error}
            </div>
          )}

          <form onSubmit={handleEmailAuth}>
            <div className="form-group">
              <label>{lang === 'en' ? 'Email' : '邮箱'}</label>
              <input
                type="email"
                className="form-input"
                placeholder={lang === 'en' ? 'Enter your email' : '请输入邮箱'}
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label>{lang === 'en' ? 'Password' : '密码'}</label>
              <input
                type="password"
                className="form-input"
                placeholder={lang === 'en' ? 'Enter your password' : '请输入密码'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <button
              type="submit"
              className="btn btn-primary"
              style={{ width: '100%' }}
              disabled={loading}
            >
              {loading
                ? (lang === 'en' ? 'Processing...' : '处理中...')
                : isSignUp
                  ? (lang === 'en' ? 'Sign Up' : '注册')
                  : (lang === 'en' ? 'Sign In' : '登录')}
            </button>
          </form>

          <div style={{
            display: 'flex', alignItems: 'center', gap: 12,
            margin: '20px 0', color: 'var(--color-text-secondary)', fontSize: 13,
          }}>
            <hr style={{ flex: 1, border: 'none', borderTop: '1px solid var(--color-border)' }} />
            <span>{lang === 'en' ? 'or' : '或'}</span>
            <hr style={{ flex: 1, border: 'none', borderTop: '1px solid var(--color-border)' }} />
          </div>

          <button
            className="btn btn-secondary"
            style={{ width: '100%' }}
            onClick={handleGoogle}
            disabled={loading}
          >
            {lang === 'en' ? 'Continue with Google' : '使用 Google 登录'}
          </button>

          <p style={{ textAlign: 'center', marginTop: 20, fontSize: 13, color: 'var(--color-text-secondary)' }}>
            {isSignUp
              ? (lang === 'en' ? 'Already have an account?' : '已有账号？')
              : (lang === 'en' ? "Don't have an account?" : '没有账号？')}
            {' '}
            <button
              onClick={() => { setIsSignUp(!isSignUp); setError('') }}
              style={{
                border: 'none', background: 'none', color: 'var(--color-primary)',
                cursor: 'pointer', fontSize: 13, fontWeight: 600,
              }}
            >
              {isSignUp ? (lang === 'en' ? 'Sign In' : '登录') : (lang === 'en' ? 'Sign Up' : '注册')}
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}
