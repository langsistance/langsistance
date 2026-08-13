/**
 * Document preview decision for document rows in the results page.
 * Pure function — the iframe/fallback/unavailable choice is derived only
 * from the row URL so it stays unit-testable.
 */

function extractInnerUrl(proxyUrl) {
  try {
    const inner = new URL(proxyUrl).searchParams.get('url')
    if (!inner) return null
    try {
      return decodeURIComponent(inner)
    } catch {
      return inner // already decoded or malformed — keep raw value
    }
  } catch {
    const match = /[?&]url=([^&]+)/.exec(proxyUrl)
    if (!match) return null
    try {
      return decodeURIComponent(match[1])
    } catch {
      return match[1]
    }
  }
}

function isPdfPath(url) {
  try {
    return new URL(url).pathname.toLowerCase().endsWith('.pdf')
  } catch {
    return url.toLowerCase().split('?')[0].endsWith('.pdf')
  }
}

// Adobe PDF open parameters honored by the browser PDF viewers — the
// thumbnail navigation pane starts collapsed (still expandable via the
// viewer's sidebar button).  Fragments never reach the server.
function withViewerParams(src) {
  return src.includes('#') ? src : `${src}#navpanes=0`
}

export function buildDocPreview(url) {
  if (!url || typeof url !== 'string') return { mode: 'unavailable' }

  if (url.includes('/uspto/download')) {
    const inner = extractInnerUrl(url)
    if (!inner) return { mode: 'fallback', url }
    return isPdfPath(inner)
      ? { mode: 'iframe', src: withViewerParams(`${url}&inline=1`) }
      : { mode: 'fallback', url }
  }

  return isPdfPath(url)
    ? { mode: 'iframe', src: withViewerParams(url) }
    : { mode: 'fallback', url }
}
