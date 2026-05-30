'use client'

import { useI18n } from '@/lib/app-i18n'
import {
  getLanguageFlagClass,
  getLanguageToggleLabel,
  getLanguageToggleTitle,
} from '@/lib/languageToggle'

export default function LanguageToggleButton() {
  const { lang, setLang } = useI18n()

  return (
    <button
      type="button"
      className="language-toggle-btn"
      onClick={() => setLang(lang === 'en' ? 'zh' : 'en')}
      title={getLanguageToggleTitle(lang)}
    >
      <span className={`language-flag ${getLanguageFlagClass(lang)}`} aria-hidden="true" />
      <span>{getLanguageToggleLabel(lang)}</span>
    </button>
  )
}
