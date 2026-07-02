'use client'

import { useState } from 'react'
import { login, signup } from '@/lib/auth-client'
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
