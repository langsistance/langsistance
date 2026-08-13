import { test } from 'node:test'
import assert from 'node:assert/strict'
import { buildDocPreview } from './docPreview.js'

const PROXY = 'https://api.copiioai.com/uspto/download'

test('proxy url with inner pdf builds iframe src with inline param', () => {
  const url = `${PROXY}?url=${encodeURIComponent('https://api.uspto.gov/api/v1/download/applications/1/file.pdf')}`
  const preview = buildDocPreview(url)
  assert.equal(preview.mode, 'iframe')
  assert.equal(preview.src, `${url}&inline=1`)
})

test('proxy url with inner docx falls back', () => {
  const url = `${PROXY}?url=${encodeURIComponent('https://api.uspto.gov/api/v1/download/applications/1/file.docx')}`
  const preview = buildDocPreview(url)
  assert.equal(preview.mode, 'fallback')
  assert.equal(preview.url, url)
})

test('plain pdf url embeds without inline param', () => {
  const preview = buildDocPreview('https://example.com/patent.pdf?token=1')
  assert.equal(preview.mode, 'iframe')
  assert.equal(preview.src, 'https://example.com/patent.pdf?token=1')
})

test('plain docx url falls back', () => {
  const preview = buildDocPreview('https://example.com/patent.docx')
  assert.equal(preview.mode, 'fallback')
})

test('empty url is unavailable', () => {
  assert.deepEqual(buildDocPreview(''), { mode: 'unavailable' })
  assert.deepEqual(buildDocPreview(null), { mode: 'unavailable' })
})

test('proxy url without inner url param falls back', () => {
  const preview = buildDocPreview(`${PROXY}`)
  assert.equal(preview.mode, 'fallback')
})
