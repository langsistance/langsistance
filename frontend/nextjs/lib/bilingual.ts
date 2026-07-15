/**
 * Client-side bilingual text (zh:...|en:...) parser.
 *
 * This is a safety-net for cases where the backend may return raw bilingual
 * text. The primary localization happens server-side in the API endpoints;
 * this utility ensures that even if raw text reaches the client, only one
 * language is displayed.
 */

const BILINGUAL_SPLIT_RE = /\|(?=(?:zh|en):)/

/**
 * Extract the text for the given language from a bilingual string.
 *
 * @param text - Text in "zh:Chinese|en:English" format, or plain text.
 * @param lang - Language code ("zh" or "en").
 * @returns The text for the requested language, with the prefix stripped.
 */
export function pickLang(text: string | null | undefined, lang: string): string {
  if (!text) return ''

  // Normalize lang to base form
  const baseLang = (lang || 'zh').split('-')[0].toLowerCase()
  const effectiveLang = baseLang === 'en' ? 'en' : 'zh'
  const otherLang = effectiveLang === 'en' ? 'zh' : 'en'

  const prefix = effectiveLang + ':'
  const otherPrefix = otherLang + ':'

  // Split only on | followed by zh: or en:
  const parts = text.split(BILINGUAL_SPLIT_RE)

  for (const part of parts) {
    const trimmed = part.trim()
    if (trimmed.startsWith(prefix)) {
      let result = trimmed.slice(prefix.length)

      // Strip embedded other-language marker if present (corrupted data)
      const otherIdx = result.indexOf(otherPrefix)
      if (otherIdx !== -1) {
        result = result.slice(0, otherIdx).trimEnd()
      }

      // Strip any re-appearance of the same prefix (double-applied)
      const sameIdx = result.indexOf(prefix)
      if (sameIdx !== -1) {
        result = result.slice(0, sameIdx).trimEnd()
      }

      return result
    }
  }

  // Fallback: single-language data — strip leading prefix if present.
  // Also truncate at embedded other-language markers (corrupted data).
  for (const pfx of ['zh:', 'en:']) {
    if (text.startsWith(pfx)) {
      let result = text.slice(pfx.length)
      const other = pfx === 'zh:' ? 'en:' : 'zh:'
      const idx = result.indexOf(other)
      if (idx !== -1) {
        result = result.slice(0, idx).trimEnd()
      }
      return result
    }
  }
  return text
}
