import test from 'node:test'
import assert from 'node:assert/strict'

import { attachImageRetryHandlers, buildImageRetryUrl, getImageRetryDelay } from './imageRetry.js'

function createImage(src) {
  const listeners = new Map()
  const classes = new Set()
  const parentNode = {
    insertedNodes: [],
    insertBefore(node, referenceNode) {
      this.insertedNodes.push({ node, referenceNode })
      node.parentNode = this
    },
    removeChild(node) {
      this.insertedNodes = this.insertedNodes.filter((entry) => entry.node !== node)
      node.parentNode = null
    },
  }

  return {
    dataset: {},
    style: {},
    src,
    parentNode,
    nextSibling: null,
    ownerDocument: {
      createElement(tagName) {
        return {
          tagName: tagName.toUpperCase(),
          className: '',
          href: '',
          textContent: '',
          target: '',
          rel: '',
          parentNode: null,
        }
      },
    },
    complete: false,
    naturalWidth: 0,
    getAttribute(name) {
      return name === 'src' ? this.src : null
    },
    addEventListener(name, handler) {
      listeners.set(name, handler)
    },
    removeEventListener(name, handler) {
      if (listeners.get(name) === handler) listeners.delete(name)
    },
    classList: {
      add(...names) {
        names.forEach((name) => classes.add(name))
      },
      remove(...names) {
        names.forEach((name) => classes.delete(name))
      },
      contains(name) {
        return classes.has(name)
      },
    },
    fire(name) {
      listeners.get(name)?.()
    },
    listenerCount() {
      return listeners.size
    },
  }
}

function rootWithImages(images) {
  return {
    querySelectorAll(selector) {
      assert.equal(selector, 'img')
      return images
    },
  }
}

test('computes exponential image retry delay with bounded jitter', () => {
  const options = { baseDelayMs: 1000, maxDelayMs: 5000, jitterRatio: 0.2 }

  assert.equal(getImageRetryDelay(1, options, () => 0), 1000)
  assert.equal(getImageRetryDelay(2, options, () => 0.5), 2200)
  assert.equal(getImageRetryDelay(5, options, () => 1), 6000)
})

test('builds retry URLs with a cache-busting attempt marker', () => {
  const retryUrl = buildImageRetryUrl('https://example.com/photo.jpg?size=large#preview', 2, () => 12345)

  assert.equal(
    retryUrl,
    'https://example.com/photo.jpg?size=large&copiio_img_retry=2-12345#preview'
  )
})

test('retries failed images before marking them failed', () => {
  const img = createImage('https://example.com/photo.jpg')
  const timers = []

  attachImageRetryHandlers(rootWithImages([img]), {
    maxRetries: 2,
    baseDelayMs: 1000,
    jitterRatio: 0,
    now: () => 111,
    setTimeoutFn(callback, delay) {
      timers.push({ callback, delay })
      return timers.length
    },
    clearTimeoutFn() {},
  })

  img.fire('error')
  assert.equal(timers[0].delay, 1000)
  assert.equal(img.dataset.imageRetryCount, '1')
  assert.equal(img.classList.contains('image-retrying'), true)

  timers[0].callback()
  assert.equal(img.src, 'https://example.com/photo.jpg?copiio_img_retry=1-111')

  img.fire('error')
  timers[1].callback()
  assert.equal(img.src, 'https://example.com/photo.jpg?copiio_img_retry=2-111')

  img.fire('error')
  assert.equal(img.classList.contains('image-load-failed'), true)
})

test('shows the original image link when retries are exhausted', () => {
  const img = createImage('https://example.com/photo.jpg?size=large')

  attachImageRetryHandlers(rootWithImages([img]), {
    maxRetries: 0,
  })

  img.fire('error')

  assert.equal(img.style.display, 'none')
  assert.equal(img.parentNode.insertedNodes.length, 1)

  const fallbackLink = img.parentNode.insertedNodes[0].node
  assert.equal(fallbackLink.tagName, 'A')
  assert.equal(fallbackLink.className, 'image-fallback-link')
  assert.equal(fallbackLink.href, 'https://example.com/photo.jpg?size=large')
  assert.equal(fallbackLink.textContent, 'https://example.com/photo.jpg?size=large')
  assert.equal(fallbackLink.target, '_blank')
  assert.equal(fallbackLink.rel, 'noopener noreferrer')
})

test('cleanup removes listeners and clears pending retry timers', () => {
  const img = createImage('https://example.com/photo.jpg')
  const cleared = []

  const cleanup = attachImageRetryHandlers(rootWithImages([img]), {
    setTimeoutFn() {
      return 42
    },
    clearTimeoutFn(timerId) {
      cleared.push(timerId)
    },
  })

  img.fire('error')
  cleanup()

  assert.deepEqual(cleared, [42])
  assert.equal(img.listenerCount(), 0)
})
