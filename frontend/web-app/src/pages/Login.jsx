import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
} from 'firebase/auth'
import { auth } from '../firebase'
import { useI18n } from '../i18n'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isSignUp, setIsSignUp] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()
  const { t, lang, setLang } = useI18n()

  async function handleEmailAuth(e) {
    e.preventDefault()
    setError('')
    try {
      if (isSignUp) {
        await createUserWithEmailAndPassword(auth, email, password)
      } else {
        await signInWithEmailAndPassword(auth, email, password)
      }
      navigate('/chat')
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleGoogle() {
    setError('')
    try {
      await signInWithPopup(auth, new GoogleAuthProvider())
      navigate('/chat')
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <img src="/app/logo.png" alt="Logo" style={{ width: 40, height: 40, borderRadius: 10, objectFit: 'contain' }} />
            <h1 style={{ margin: 0 }}>CopiioAI</h1>
          </div>
          <button
            className="language-toggle-btn"
            onClick={() => setLang(lang === 'en' ? 'zh' : 'en')}
          >
            <span>{lang === 'en' ? '🇺🇸' : '🇨🇳'}</span>
            <span>{lang === 'en' ? 'English' : '中文'}</span>
          </button>
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
          <button type="submit" className="btn btn-primary" style={{ width: '100%' }}>
            {isSignUp ? (lang === 'en' ? 'Sign Up' : '注册') : (lang === 'en' ? 'Sign In' : '登录')}
          </button>
        </form>

        <div className="login-divider">
          <hr />
          <span>{lang === 'en' ? 'or' : '或'}</span>
          <hr />
        </div>

        <button className="btn btn-secondary" style={{ width: '100%' }} onClick={handleGoogle}>
          {lang === 'en' ? 'Continue with Google' : '使用 Google 登录'}
        </button>

        <p className="login-footer">
          {isSignUp
            ? (lang === 'en' ? 'Already have an account?' : '已有账号？')
            : (lang === 'en' ? "Don't have an account?" : '没有账号？')}
          <button onClick={() => setIsSignUp(!isSignUp)}>
            {isSignUp ? (lang === 'en' ? 'Sign In' : '登录') : (lang === 'en' ? 'Sign Up' : '注册')}
          </button>
        </p>
      </div>
    </div>
  )
}
