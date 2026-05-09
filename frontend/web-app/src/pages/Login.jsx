import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
} from 'firebase/auth'
import { auth } from '../firebase'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isSignUp, setIsSignUp] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

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
    <div className="min-h-screen bg-slate-900 flex items-center justify-center">
      <div className="w-full max-w-sm bg-slate-800 rounded-xl p-8 border border-slate-700">
        <h1 className="text-2xl font-bold text-white mb-2">CopiioAI</h1>
        <p className="text-slate-400 text-sm mb-6">{isSignUp ? '创建账号' : '登录你的账号'}</p>

        {error && <p className="text-red-400 text-sm mb-4">{error}</p>}

        <form onSubmit={handleEmailAuth} className="flex flex-col gap-3">
          <input
            type="email"
            placeholder="邮箱"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-teal-500"
            required
          />
          <input
            type="password"
            placeholder="密码"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-teal-500"
            required
          />
          <button
            type="submit"
            className="bg-teal-600 hover:bg-teal-500 text-white text-sm font-medium py-2 rounded-lg transition-colors"
          >
            {isSignUp ? '注册' : '登录'}
          </button>
        </form>

        <div className="flex items-center gap-3 my-4">
          <div className="flex-1 h-px bg-slate-700" />
          <span className="text-slate-500 text-xs">或</span>
          <div className="flex-1 h-px bg-slate-700" />
        </div>

        <button
          onClick={handleGoogle}
          className="w-full border border-slate-600 hover:border-slate-500 text-slate-300 text-sm py-2 rounded-lg transition-colors"
        >
          使用 Google 登录
        </button>

        <p className="text-center text-slate-500 text-xs mt-4">
          {isSignUp ? '已有账号？' : '没有账号？'}
          <button
            onClick={() => setIsSignUp(!isSignUp)}
            className="text-teal-400 ml-1 hover:underline"
          >
            {isSignUp ? '登录' : '注册'}
          </button>
        </p>
      </div>
    </div>
  )
}
