import test from 'node:test'
import assert from 'node:assert/strict'

import { isImageUrl, renderMarkdownToHtml } from './markdownRender.js'
// looksLikeBareUrl is not exported — test via renderMarkdownToHtml behaviour

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

test('renders bare-domain URL inside backtick codespan as clickable link', () => {
  // Simulates the user's actual output:
  // **文件路径**：`pt.cnipr.com/static/.../CN2021113253961A.PDF`
  const html = renderMarkdownToHtml(
    '**文件路径**：`pt.cnipr.com/static/a8ba192e794358b0997c244c81602bab/78FF4A/D82913/1463451601/1782098787/1200/pi11/PAT/151/35/CN2021113253961A_20210813/CN2021113253961A.PDF`'
  )

  assert.match(html, /<a\b/)
  assert.match(html, /href="https:\/\/pt\.cnipr\.com/)
  assert.match(html, /target="_blank"/)
  assert.match(html, /rel="noopener\s+noreferrer"/)
  assert.doesNotMatch(html, /<code>/)
})

test('adds target=_blank and rel=noopener noreferrer to markdown links', () => {
  const html = renderMarkdownToHtml('[Patent](https://pt.cnipr.com/static/file.PDF)')

  assert.match(html, /<a\b/)
  assert.match(html, /href="https:\/\/pt\.cnipr\.com\/static\/file\.PDF"/)
  assert.match(html, /target="_blank"/)
  assert.match(html, /rel="noopener\s+noreferrer"/)
})

test('keeps regular inline code as code when it is not a URL', () => {
  const html = renderMarkdownToHtml('Use `const x = 1` here and `npm install foo`')

  assert.match(html, /<code>const x = 1<\/code>/)
  assert.match(html, /<code>npm install foo<\/code>/)
  assert.doesNotMatch(html, /<a\b/)
})

test('keeps codespan that looks like a domain but has no path as code', () => {
  // "foo.bar" has a TLD but no path — not link-worthy
  const html = renderMarkdownToHtml('the domain `foo.bar` is not a link')
  assert.doesNotMatch(html, /<a\b/)
  assert.match(html, /<code>foo\.bar<\/code>/)
})

test('renders bare URL with existing https:// as clickable link with protocol preserved', () => {
  const html = renderMarkdownToHtml('`https://example.com/path/to/file.pdf`')

  assert.match(html, /<a\b/)
  assert.match(html, /href="https:\/\/example\.com\/path\/to\/file\.pdf"/)
  assert.doesNotMatch(html, /<code>/)
})

test('renders bare URL without protocol by adding https://', () => {
  const html = renderMarkdownToHtml('`cdn.example.com/files/report.pdf`')

  assert.match(html, /<a\b/)
  assert.match(html, /href="https:\/\/cdn\.example\.com\/files\/report\.pdf"/)
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
