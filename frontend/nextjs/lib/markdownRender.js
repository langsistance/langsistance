import { marked } from 'marked'
import { markedHighlight } from 'marked-highlight'
import hljs from 'highlight.js'

const IMAGE_URL_RE = /(?:\.(?:apng|avif|gif|jpe?g|png|svg|webp)|@(?:apng|avif|gif|jpe?g|png|svg|webp))(?:[?#].*)?$/i
const EXTENSIONLESS_IMAGE_PATHS = [
  { host: 'i.scdn.co', pathPrefix: '/image/' },
]

let configured = false

function escapeAttribute(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
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
        if (isImageUrl(token.href)) {
          const altText = token.text || 'image'
          return `<img src="${escapeAttribute(token.href)}" alt="${escapeAttribute(altText)}">`
        }
        const title = token.title ? ` title="${escapeAttribute(token.title)}"` : ''
        return `<a href="${escapeAttribute(token.href)}"${title}>${escapeAttribute(token.text)}</a>`
      },
    },
  })
  configured = true
}

export function renderMarkdownToHtml(content) {
  configureMarked()
  return marked.parse(content || '')
}
