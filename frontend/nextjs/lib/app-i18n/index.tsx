'use client'
import { createContext, useContext, useCallback, useMemo, useState } from 'react'
import en from './locales/en'
import zh from './locales/zh'

type Locale = typeof en
const LOCALES: Record<string, Locale> = { en, zh }
const STORAGE_KEY = 'apiforge-language'

function getInitialLang(): string {
  if (typeof window === 'undefined') return 'en'
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved && LOCALES[saved]) return saved
  } catch {}
  return 'en'
}

interface I18nContextValue {
  lang: string
  setLang: (lang: string) => void
  t: (key: string, params?: Record<string, string | number>) => string
}

const I18nContext = createContext<I18nContextValue | null>(null)

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = useState(getInitialLang)

  const setLang = useCallback((newLang: string) => {
    if (!LOCALES[newLang]) return
    setLangState(newLang)
    try { localStorage.setItem(STORAGE_KEY, newLang) } catch {}
  }, [])

  const t = useCallback((key: string, params: Record<string, string | number> = {}): string => {
    const locale = LOCALES[lang] as Record<string, unknown>
    const keys = key.split('.')
    let value: unknown = locale
    for (const k of keys) {
      if (value && typeof value === 'object' && k in (value as object)) {
        value = (value as Record<string, unknown>)[k]
      } else {
        return key
      }
    }
    if (typeof value !== 'string') return key
    return Object.entries(params).reduce(
      (str, [k, v]) => str.replace(new RegExp(`\\{${k}\\}`, 'g'), String(v)),
      value
    )
  }, [lang])

  const value = useMemo(() => ({ lang, setLang, t }), [lang, setLang, t])

  return (
    <I18nContext.Provider value={value}>
      {children}
    </I18nContext.Provider>
  )
}

export function useI18n() {
  const ctx = useContext(I18nContext)
  if (!ctx) throw new Error('useI18n must be used within I18nProvider')
  return ctx
}
