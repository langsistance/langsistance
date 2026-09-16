import test from 'node:test'
import assert from 'node:assert/strict'

import { TRUST_COPY, TRUST_NOTES } from './trustNotice.js'

const LANGS = ['zh', 'en']

test('两种语言都提供 headline 与 more', () => {
  for (const lang of LANGS) {
    assert.equal(typeof TRUST_COPY[lang].headline, 'string', `${lang} headline`)
    assert.ok(TRUST_COPY[lang].headline.trim().length > 0, `${lang} headline 非空`)
    assert.ok(TRUST_COPY[lang].more.trim().length > 0, `${lang} more 非空`)
  }
})

test('两种语言的承诺条数与 key 集合完全一致', () => {
  const keysOf = (lang) => TRUST_COPY[lang].items.map((i) => i.key).sort()
  assert.deepEqual(keysOf('zh'), keysOf('en'))
  assert.equal(TRUST_COPY.zh.items.length, 3)
})

test('每条承诺的 key / label / desc 均为非空字符串', () => {
  for (const lang of LANGS) {
    for (const item of TRUST_COPY[lang].items) {
      assert.ok(item.key.trim().length > 0, `${lang} key`)
      assert.ok(item.label.trim().length > 0, `${lang} ${item.key} label`)
      assert.ok(item.desc.trim().length > 0, `${lang} ${item.key} desc`)
    }
  }
})

test('两种语言都提供 login 与 focus 短句', () => {
  for (const lang of LANGS) {
    assert.ok(TRUST_NOTES[lang].login.trim().length > 0, `${lang} login`)
    assert.ok(TRUST_NOTES[lang].focus.trim().length > 0, `${lang} focus`)
  }
})
