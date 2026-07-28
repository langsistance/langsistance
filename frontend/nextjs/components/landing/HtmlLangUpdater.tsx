'use client'

import { useEffect } from 'react'
import { useLandingI18n, type LangKey } from '@/lib/landing-i18n'

const LANG_TO_HTML_LANG: Record<LangKey, string> = {
  en: 'en',
  zh: 'zh-Hans',
  ja: 'ja',
  ko: 'ko',
  es: 'es',
  fr: 'fr',
  de: 'de',
}

/**
 * Updates the <html lang="..."> attribute whenever the user switches language.
 * Place this inside the LandingI18nProvider tree.
 */
export default function HtmlLangUpdater() {
  const { lang } = useLandingI18n()

  useEffect(() => {
    document.documentElement.lang = LANG_TO_HTML_LANG[lang] || 'en'
  }, [lang])

  return null
}
