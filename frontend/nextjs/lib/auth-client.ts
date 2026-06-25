/**
 * 通过后端代理与 Firebase Auth 交互。
 *
 * 国内浏览器无法直连 *.googleapis.com，所以登录/刷新等流程都走后端代理。
 * 拿到的 idToken 是 Firebase 真签的，可被现有 verify_firebase_token 校验。
 *
 * Token 存 localStorage；getValidToken 自动在过期前 60s 触发 refresh。
 * 提供轻量订阅机制替代 Firebase 的 onAuthStateChanged。
 *
 * 密码在客户端用 AES-GCM 加密后再传输，后端解密后转发 Firebase。
 * 密码明文不会出现在 Network 面板的请求体里。
 */

const STORAGE_KEY = 'cp_auth_v1'
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'https://api.copiioai.com'

/** 与后端 PASSWORD_ENCRYPTION_SECRET 保持一致 */
const PASSWORD_PEPPER =
  process.env.NEXT_PUBLIC_PASSWORD_PEPPER || 'copiioai-default-pepper-key-2024'

/**
 * 用 AES-GCM + PBKDF2 加密密码，返回 salt+iv+ciphertext 的 base64 串。
 * 盐值和 IV 均为随机生成，同一密码每次加密结果不同。
 */
async function encryptPassword(password: string): Promise<string> {
  const enc = new TextEncoder()
  const salt = crypto.getRandomValues(new Uint8Array(16))
  const iv = crypto.getRandomValues(new Uint8Array(12))

  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    enc.encode(PASSWORD_PEPPER),
    'PBKDF2',
    false,
    ['deriveKey'],
  )
  const key = await crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt, iterations: 100_000, hash: 'SHA-256' },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt'],
  )

  const ciphertext = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    key,
    enc.encode(password),
  )

  // concat salt(16) + iv(12) + ciphertext → base64
  const buf = new Uint8Array(16 + 12 + ciphertext.byteLength)
  buf.set(salt, 0)
  buf.set(iv, 16)
  buf.set(new Uint8Array(ciphertext), 28)

  let binary = ''
  for (let i = 0; i < buf.byteLength; i++) {
    binary += String.fromCharCode(buf[i])
  }
  return btoa(binary)
}

export interface AuthUser {
  uid: string
  email: string
}

interface StoredAuth {
  idToken: string
  refreshToken: string
  expiresAt: number // ms epoch
  uid: string
  email: string
}

type Listener = (user: AuthUser | null) => void
const listeners = new Set<Listener>()

function isBrowser() {
  return typeof window !== 'undefined'
}

function load(): StoredAuth | null {
  if (!isBrowser()) return null
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    return raw ? (JSON.parse(raw) as StoredAuth) : null
  } catch {
    return null
  }
}

function save(s: StoredAuth | null) {
  if (!isBrowser()) return
  if (s) window.localStorage.setItem(STORAGE_KEY, JSON.stringify(s))
  else window.localStorage.removeItem(STORAGE_KEY)
  notify(s ? { uid: s.uid, email: s.email } : null)
}

function notify(user: AuthUser | null) {
  listeners.forEach((cb) => {
    try {
      cb(user)
    } catch {
      /* swallow */
    }
  })
}

if (isBrowser()) {
  window.addEventListener('storage', (e) => {
    if (e.key !== STORAGE_KEY) return
    const s = load()
    notify(s ? { uid: s.uid, email: s.email } : null)
  })
}

async function call<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`${path} ${res.status}${text ? ` — ${text}` : ''}`)
  }
  return res.json() as Promise<T>
}

interface AuthResponse {
  idToken: string
  refreshToken: string
  expiresIn: number
  localId: string
  email?: string
}

function persist(r: AuthResponse, email: string) {
  const stored: StoredAuth = {
    idToken: r.idToken,
    refreshToken: r.refreshToken,
    expiresAt: Date.now() + r.expiresIn * 1000,
    uid: r.localId,
    email: r.email ?? email,
  }
  save(stored)
  return stored
}

export async function login(email: string, password: string): Promise<AuthUser> {
  const encryptedPassword = await encryptPassword(password)
  const r = await call<AuthResponse>('/auth/login', { email, encryptedPassword })
  const s = persist(r, email)
  return { uid: s.uid, email: s.email }
}

export async function signup(email: string, password: string): Promise<AuthUser> {
  const encryptedPassword = await encryptPassword(password)
  const r = await call<AuthResponse>('/auth/signup', { email, encryptedPassword })
  const s = persist(r, email)
  return { uid: s.uid, email: s.email }
}

export async function resetPassword(email: string): Promise<void> {
  await call<{ ok: boolean }>('/auth/reset', { email })
}

export function logout(): void {
  save(null)
}

let refreshing: Promise<string | null> | null = null

export async function getValidToken(): Promise<string | null> {
  const s = load()
  if (!s) return null
  if (Date.now() < s.expiresAt - 60_000) return s.idToken

  if (!refreshing) {
    refreshing = (async () => {
      try {
        const r = await call<AuthResponse>('/auth/refresh', { refreshToken: s.refreshToken })
        const updated = persist(r, s.email)
        return updated.idToken
      } catch {
        save(null)
        return null
      } finally {
        refreshing = null
      }
    })()
  }
  return refreshing
}

export function getCurrentUser(): AuthUser | null {
  const s = load()
  return s ? { uid: s.uid, email: s.email } : null
}

export function onAuthChange(cb: Listener): () => void {
  listeners.add(cb)
  // fire current state synchronously (matches Firebase semantics: subscribe gets initial)
  queueMicrotask(() => cb(getCurrentUser()))
  return () => {
    listeners.delete(cb)
  }
}
