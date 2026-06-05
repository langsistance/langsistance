const DEFAULT_OPTIONS = {
  maxRetries: 3,
  baseDelayMs: 1000,
  maxDelayMs: 8000,
  jitterRatio: 0.2,
  cacheBustParam: 'copiio_img_retry',
}

function getOptions(options = {}) {
  return { ...DEFAULT_OPTIONS, ...options }
}

export function getImageRetryDelay(attempt, options = {}, random = Math.random) {
  const resolved = getOptions(options)
  const exponentialDelay = resolved.baseDelayMs * 2 ** Math.max(0, attempt - 1)
  const cappedDelay = Math.min(exponentialDelay, resolved.maxDelayMs)
  const jitter = cappedDelay * resolved.jitterRatio * random()
  return Math.round(cappedDelay + jitter)
}

export function buildImageRetryUrl(src, attempt, now = Date.now, options = {}) {
  const resolved = getOptions(options)
  const marker = `${attempt}-${now()}`

  try {
    const base = typeof window !== 'undefined' ? window.location.href : 'http://localhost'
    const retryUrl = new URL(src, base)
    retryUrl.searchParams.set(resolved.cacheBustParam, marker)
    return retryUrl.toString()
  } catch {
    const separator = src.includes('?') ? '&' : '?'
    return `${src}${separator}${resolved.cacheBustParam}=${encodeURIComponent(marker)}`
  }
}

function showImageFallbackLink(img, originalSrc) {
  if (!img || !originalSrc || img.dataset?.imageFallbackAttached === 'true') return null

  const doc = img.ownerDocument
  const parent = img.parentNode
  if (!doc?.createElement || !parent?.insertBefore) return null

  const link = doc.createElement('a')
  link.className = 'image-fallback-link'
  link.href = originalSrc
  link.textContent = originalSrc
  link.target = '_blank'
  link.rel = 'noopener noreferrer'

  img.dataset.imageFallbackAttached = 'true'
  img.style.display = 'none'
  parent.insertBefore(link, img.nextSibling)
  return link
}

export function attachImageRetryHandlers(root, options = {}) {
  if (!root?.querySelectorAll) return () => {}

  const resolved = getOptions(options)
  const setTimer = resolved.setTimeoutFn || setTimeout
  const clearTimer = resolved.clearTimeoutFn || clearTimeout
  const now = resolved.now || Date.now
  const random = resolved.random || Math.random
  const timers = new Set()
  const cleanups = []
  const fallbackLinks = new Set()
  const images = Array.from(root.querySelectorAll('img'))

  images.forEach((img) => {
    if (!img || img.dataset?.imageRetryAttached === 'true') return

    const originalSrc = img.getAttribute?.('src') || img.src
    if (!originalSrc) return

    img.dataset.imageRetryAttached = 'true'
    img.dataset.imageRetryOriginalSrc = originalSrc
    img.dataset.imageRetryCount = '0'
    img.classList?.add('image-loading')

    const handleLoad = () => {
      img.classList?.remove('image-loading', 'image-retrying', 'image-load-failed')
      img.dataset.imageRetryLoaded = 'true'
      img.style.opacity = '1'
    }

    const handleError = () => {
      const nextAttempt = Number(img.dataset.imageRetryCount || '0') + 1

      if (nextAttempt > resolved.maxRetries) {
        img.classList?.remove('image-loading', 'image-retrying')
        img.classList?.add('image-load-failed')
        img.style.opacity = '1'
        const fallbackLink = showImageFallbackLink(img, originalSrc)
        if (fallbackLink) fallbackLinks.add(fallbackLink)
        resolved.onFinalError?.(img)
        return
      }

      img.dataset.imageRetryCount = String(nextAttempt)
      img.classList?.remove('image-loading')
      img.classList?.add('image-retrying')

      const delay = getImageRetryDelay(nextAttempt, resolved, random)
      const timerId = setTimer(() => {
        timers.delete(timerId)
        img.src = buildImageRetryUrl(originalSrc, nextAttempt, now, resolved)
      }, delay)
      timers.add(timerId)
    }

    img.addEventListener('load', handleLoad)
    img.addEventListener('error', handleError)

    if (img.complete && img.naturalWidth > 0) {
      handleLoad()
    }

    cleanups.push(() => {
      img.removeEventListener('load', handleLoad)
      img.removeEventListener('error', handleError)
    })
  })

  return () => {
    timers.forEach((timerId) => clearTimer(timerId))
    timers.clear()
    fallbackLinks.forEach((link) => {
      if (link.parentNode?.removeChild) link.parentNode.removeChild(link)
    })
    fallbackLinks.clear()
    cleanups.forEach((cleanup) => cleanup())
  }
}
