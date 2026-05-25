import test from 'node:test'
import assert from 'node:assert/strict'

import { isImageUrl, renderMarkdownToHtml } from './markdownRender.js'

test('detects image URLs even when they include query strings', () => {
  assert.equal(isImageUrl('https://example.com/photo.jpg?size=large'), true)
  assert.equal(isImageUrl('https://example.com/photo.webp#preview'), true)
  assert.equal(isImageUrl('https://example.com/file.pdf'), false)
})

test('detects Decrypt proxy image URLs that end with an at-format suffix', () => {
  assert.equal(
    isImageUrl('https://img.decrypt.co/insecure/rs:fill:1024:512:1:0/plain/https://cdn.decrypt.co/wp-content/uploads/2026/04/decrypt-style-moonpay-gID_7.png@png'),
    true
  )
})

test('detects Spotify CDN image URLs that do not include file extensions', () => {
  assert.equal(
    isImageUrl('https://i.scdn.co/image/ab676161000051744293385d324db8558179afd9'),
    true
  )
})

test('renders markdown links that point to images as img tags', () => {
  const html = renderMarkdownToHtml('[Preview](https://example.com/photo.jpg?size=large)')

  assert.match(html, /<img\b/)
  assert.match(html, /src="https:\/\/example\.com\/photo\.jpg\?size=large"/)
  assert.match(html, /alt="Preview"/)
  assert.doesNotMatch(html, /<a\b/)
})

test('keeps non-image markdown links as links', () => {
  const html = renderMarkdownToHtml('[Open PDF](https://example.com/file.pdf)')

  assert.match(html, /<a\b/)
  assert.match(html, /href="https:\/\/example\.com\/file\.pdf"/)
  assert.doesNotMatch(html, /<img\b/)
})

test('renders plain image URL fields as inline images', () => {
  const imageUrl = 'https://img.decrypt.co/insecure/rs:fill:1024:512:1:0/plain/https://cdn.decrypt.co/wp-content/uploads/2026/04/decrypt-style-moonpay-gID_7.png@png'
  const html = renderMarkdownToHtml(`**Image URL:** ${imageUrl}`)

  assert.match(html, /<img\b/)
  assert.match(html, new RegExp(`src="${imageUrl.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"`))
  assert.doesNotMatch(html, /Image URL:<\/strong> https:/)
})

test('renders Spotify display image fields as inline images', () => {
  const imageUrl = 'https://i.scdn.co/image/ab676161000051744293385d324db8558179afd9'
  const html = renderMarkdownToHtml(`**Display image:** ${imageUrl}`)

  assert.match(html, /<img\b/)
  assert.match(html, new RegExp(`src="${imageUrl.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"`))
  assert.doesNotMatch(html, /Display image:<\/strong> https:/)
})
