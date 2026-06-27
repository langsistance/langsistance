import { marked } from 'marked'
import { markedHighlight } from 'marked-highlight'
import hljs from 'highlight.js'

const IMAGE_URL_RE = /(?:\.(?:apng|avif|gif|jpe?g|png|svg|webp)|@(?:apng|avif|gif|jpe?g|png|svg|webp))(?:[?#].*)?$/i
const EXTENSIONLESS_IMAGE_PATHS = [
  { host: 'i.scdn.co', pathPrefix: '/image/' },
]

// Matches bare domain-path strings that look like URLs but are missing the
// protocol prefix — e.g. "pt.cnipr.com/static/.../file.PDF".  Requires a
// TLD of 2+ letters **and** a path segment so short domain-only tokens like
// "foo.bar" are not captured.
const BARE_URL_RE = /^(?:https?:\/\/)?[a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z0-9][-a-zA-Z0-9]*)*\.[a-zA-Z]{2,}\/[^\s<>[\]()]+$/i

let configured = false

function escapeAttribute(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}

function looksLikeBareUrl(text) {
  if (typeof text !== 'string') return false
  const trimmed = text.trim()
  if (!trimmed) return false
  return BARE_URL_RE.test(trimmed)
}

export function isImageUrl(value) {
  if (typeof value !== 'string') return false
  const url = value.trim()
  if (IMAGE_URL_RE.test(url)) return true

  try {
    const parsed = new URL(url)
    const host = parsed.hostname.toLowerCase()
    return EXTENSIONLESS_IMAGE_PATHS.some(
      ({ host: imageHost, pathPrefix }) =>
        host === imageHost && parsed.pathname.startsWith(pathPrefix)
    )
  } catch {
    return false
  }
}

function configureMarked() {
  if (configured) return

  marked.use({ gfm: true, breaks: true })
  marked.use(
    markedHighlight({
      langPrefix: 'hljs language-',
      highlight(code, lang) {
        const language = hljs.getLanguage(lang) ? lang : 'plaintext'
        return hljs.highlight(code, { language }).value
      },
    })
  )
  marked.use({
    renderer: {
      link(token) {
        // Resolve bare domain-path URLs (e.g. "res.cnipr.com/path/file.pdf")
        // so the browser does not treat them as relative paths on our origin.
        let href = token.href
        if (looksLikeBareUrl(href)) {
          href = /^https?:\/\//i.test(href) ? href : `https://${href}`
        }
        if (isImageUrl(href)) {
          const altText = token.text || 'image'
          return `<img src="${escapeAttribute(href)}" alt="${escapeAttribute(altText)}">`
        }
        const title = token.title ? ` title="${escapeAttribute(token.title)}"` : ''
        return `<a href="${escapeAttribute(href)}"${title} target="_blank" rel="noopener noreferrer">${escapeAttribute(token.text)}</a>`
      },
      codespan(token) {
        const text = token.text || ''
        // When inline-code content is a bare URL (e.g.
        // `pt.cnipr.com/static/.../file.PDF`) render it as a clickable
        // link so users don't have to copy-paste it.
        if (looksLikeBareUrl(text)) {
          const href = /^https?:\/\//i.test(text) ? text : `https://${text}`
          return `<a href="${escapeAttribute(href)}" target="_blank" rel="noopener noreferrer">${escapeAttribute(text)}</a>`
        }
        return `<code>${escapeAttribute(text)}</code>`
      },
    },
  })
  configured = true
}

export function renderMarkdownToHtml(content) {
  configureMarked()
  return marked.parse(content || '')
}
