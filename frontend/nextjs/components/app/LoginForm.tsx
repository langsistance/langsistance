'use client'

import { useState } from 'react'
import { login, signup, loginWithGoogle } from '@/lib/auth-client'
import { validateSignupPasswordConfirmation } from '@/lib/authValidation'
import { useAuth } from '@/contexts/AuthContext'
import { useI18n } from '@/lib/app-i18n'
import LanguageToggleButton from '@/components/app/LanguageToggleButton'

/**
 * Extract a clean auth error code from various error formats:
 *   - Firebase code: "INVALID_PASSWORD"
 *   - Old proxy format: "/auth/login 400 — {"detail":"INVALID_PASSWORD"}"
 *   - FastAPI JSON: '{"detail":"INVALID_PASSWORD"}'
 * Returns the original string if no known pattern matches.
 */
function extractAuthErrorCode(raw: string): string {
  // Strip old proxy prefix: "/auth/login 400 — ..."
  let cleaned = raw.replace(/^\/auth\/\w+\s+\d{3}\s*(—|-)\s*/i, '')
  // Try to parse as JSON and extract detail
  try {
    const parsed = JSON.parse(cleaned)
    if (parsed.detail && typeof parsed.detail === 'string') {
      return parsed.detail
    }
  } catch {
    // Not JSON, use as-is
  }
  return cleaned
}

export default function LoginForm() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [isSignUp, setIsSignUp] = useState(false)
  const [error, setError] = useState('')
  const [googleLoading, setGoogleLoading] = useState(false)
  const { user } = useAuth()
  const { t, lang } = useI18n()

  async function handleEmailAuth(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    try {
      if (isSignUp) {
        const validationError = validateSignupPasswordConfirmation(password, confirmPassword, lang)
        if (validationError) {
          setError(validationError)
          return
        }
        await signup(email, password)
      } else {
        await login(email, password)
      }
      // Auth state change will trigger parent re-render — no redirect needed
    } catch (err: unknown) {
      const raw = err instanceof Error ? err.message : 'AUTH_ERROR'
      const code = extractAuthErrorCode(raw)
      // Try i18n translation first, fall back to the raw code
      const translated = t(`auth.errors.${code}`)
      setError(translated === `auth.errors.${code}` ? code : translated)
    }
  }

  async function handleGoogleSignIn() {
    setError('')
    setGoogleLoading(true)
    try {
      await loginWithGoogle()
      // Auth state change will trigger parent re-render
    } catch (err: unknown) {
      // Firebase popup closed by user — not an error worth showing
      if (err instanceof Error && (
        err.message.includes('auth/popup-closed-by-user') ||
        err.message.includes('auth/cancelled-popup-request')
      )) {
        return
      }
      const raw = err instanceof Error ? err.message : 'AUTH_ERROR'
      const code = extractAuthErrorCode(raw)
      const translated = t(`auth.errors.${code}`)
      setError(translated === `auth.errors.${code}` ? code : translated)
    } finally {
      setGoogleLoading(false)
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <img src="/logo.png" alt="Logo" style={{ width: 40, height: 40, borderRadius: 10, objectFit: 'contain' }} />
            <h1 style={{ margin: 0 }}>CopiioAI</h1>
          </div>
          <LanguageToggleButton />
        </div>

        <p className="subtitle">{isSignUp ? t('common.confirm') : t('app.description')}</p>

        {error && <div className="login-error">{error}</div>}

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
          {isSignUp && (
            <div className="form-group">
              <label>{lang === 'en' ? 'Confirm Password' : '确认密码'}</label>
              <input
                type="password"
                className="form-input"
                placeholder={lang === 'en' ? 'Enter your password again' : '请再次输入密码'}
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
              />
            </div>
          )}
          <button type="submit" className="btn btn-primary" style={{ width: '100%' }}>
            {isSignUp ? (lang === 'en' ? 'Sign Up' : '注册') : (lang === 'en' ? 'Sign In' : '登录')}
          </button>
        </form>

        {!isSignUp && (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, margin: '20px 0' }}>
              <div style={{ flex: 1, height: 1, background: 'var(--border-color, #e5e7eb)' }} />
              <span style={{ fontSize: 13, color: 'var(--text-secondary, #6b7280)' }}>
                {lang === 'en' ? 'or' : '或'}
              </span>
              <div style={{ flex: 1, height: 1, background: 'var(--border-color, #e5e7eb)' }} />
            </div>

            <button
              type="button"
              className="btn"
              onClick={handleGoogleSignIn}
              disabled={googleLoading}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 10,
                background: 'var(--surface-color, #fff)',
                border: '1px solid var(--border-color, #d1d5db)',
                color: 'var(--text-primary, #111827)',
                padding: '10px 16px',
                borderRadius: 8,
                fontSize: 14,
                fontWeight: 500,
                cursor: 'pointer',
              }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" />
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
              </svg>
              {googleLoading
                ? (lang === 'en' ? 'Signing in...' : '登录中...')
                : (lang === 'en' ? 'Continue with Google' : '使用 Google 登录')
              }
            </button>
          </>
        )}

        <p className="login-footer">
          {isSignUp
            ? (lang === 'en' ? 'Already have an account?' : '已有账号？')
            : (lang === 'en' ? "Don't have an account?" : '没有账号？')}
          <button onClick={() => {
            setIsSignUp(!isSignUp)
            setConfirmPassword('')
            setError('')
          }}>
            {isSignUp ? (lang === 'en' ? 'Sign In' : '登录') : (lang === 'en' ? 'Sign Up' : '注册')}
          </button>
        </p>
      </div>
    </div>
  )
}
