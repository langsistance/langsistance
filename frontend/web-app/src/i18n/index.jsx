import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import en from './locales/en'

const LOCALES = {
  en,
  zh: en,
}

const STORAGE_KEY = 'apiforge-language'
const I18nContext = createContext(null)

function getInitialLang() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved && LOCALES[saved]) return saved
  } catch {}
  return 'en'
}

export function I18nProvider({ children }) {
  const [lang, setLangState] = useState(getInitialLang)

  const setLang = useCallback((newLang) => {
    if (!LOCALES[newLang]) return
    setLangState(newLang)
    try { localStorage.setItem(STORAGE_KEY, newLang) } catch {}
  }, [])

  const t = useCallback((key, params = {}) => {
    const keys = key.split('.')
    let value = LOCALES[lang]

    for (const item of keys) {
      if (value && typeof value === 'object' && item in value) {
        value = value[item]
      } else {
        return key
      }
    }

    if (typeof value !== 'string') return key
    return Object.entries(params).reduce(
      (text, [param, replacement]) => text.replace(new RegExp(`\\{${param}\\}`, 'g'), String(replacement)),
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
  const context = useContext(I18nContext)
  if (!context) throw new Error('useI18n must be used within I18nProvider')
  return context
}
