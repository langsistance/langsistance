export async function copyTextToClipboard(text, options = {}) {
  const navigatorRef = options.navigatorRef ?? (
    typeof navigator !== 'undefined' ? navigator : undefined
  )
  const documentRef = options.documentRef ?? (
    typeof document !== 'undefined' ? document : undefined
  )

  if (navigatorRef?.clipboard?.writeText) {
    try {
      await navigatorRef.clipboard.writeText(text)
      return true
    } catch {
      // Public HTTP pages often block navigator.clipboard; try the legacy path.
    }
  }

  if (!documentRef?.body || !documentRef.createElement || !documentRef.execCommand) {
    return false
  }

  const textarea = documentRef.createElement('textarea')
  textarea.value = text
  textarea.setAttribute('readonly', '')
  textarea.style.position = 'fixed'
  textarea.style.top = '-1000px'
  textarea.style.left = '-1000px'
  textarea.style.opacity = '0'

  documentRef.body.appendChild(textarea)
  textarea.select()

  try {
    return documentRef.execCommand('copy')
  } catch {
    return false
  } finally {
    documentRef.body.removeChild(textarea)
  }
}
