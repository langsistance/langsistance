export function getLanguageFlagClass(lang = 'en') {
  return lang === 'zh' ? 'language-flag-cn' : 'language-flag-us'
}

export function getLanguageToggleLabel(lang = 'en') {
  return lang === 'zh' ? '中文' : 'English'
}

export function getLanguageToggleTitle(lang = 'en') {
  return lang === 'en' ? 'Switch to 中文' : 'Switch to English'
}
