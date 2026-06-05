import test from 'node:test'
import assert from 'node:assert/strict'

import {
  BILIBILI_USPTO_VIDEO_IDS,
  LANDING_HEADER_ACTIONS,
  LANDING_EXAMPLES,
  getBilibiliEmbedUrl,
  getLandingExampleBySlug,
} from './landingExamples.js'

test('landing header actions expose examples without download app buttons', () => {
  const labelKeys = LANDING_HEADER_ACTIONS.map((action) => action.labelKey)

  assert.deepEqual(labelKeys, ['header.docs', 'header.examples'])
  assert.equal(LANDING_HEADER_ACTIONS.some((action) => action.fallbackLabel === 'Chrome Extension'), false)
  assert.equal(LANDING_HEADER_ACTIONS.some((action) => action.fallbackLabel === 'Desktop App Coming Soon'), false)
})

test('USPTO China example is available as the first landing example', () => {
  const [example] = LANDING_EXAMPLES

  assert.equal(example.slug, 'uspto-china')
  assert.equal(example.href, '/examples/uspto-china')
  assert.equal(example.titleKey, 'examples.usptoChina.title')
  assert.equal(example.summaryKey, 'examples.usptoChina.summary')
  assert.match(example.title, /USPTO/)
  assert.equal(getLandingExampleBySlug('uspto-china'), example)
})

test('Bilibili embed URLs are generated from BV ids without autoplay', () => {
  assert.deepEqual(BILIBILI_USPTO_VIDEO_IDS, ['BV1p2Vz6HEjZ', 'BV1PCVz6WEPW'])
  assert.equal(
    getBilibiliEmbedUrl('BV1p2Vz6HEjZ'),
    'https://player.bilibili.com/player.html?bvid=BV1p2Vz6HEjZ&page=1&high_quality=1&autoplay=0'
  )
})
