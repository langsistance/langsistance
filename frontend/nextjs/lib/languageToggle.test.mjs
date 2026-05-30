import test from 'node:test'
import assert from 'node:assert/strict'

import {
  getLanguageFlagClass,
  getLanguageToggleLabel,
  getLanguageToggleTitle,
} from './languageToggle.js'

test('maps English to the US flag class and label', () => {
  assert.equal(getLanguageFlagClass('en'), 'language-flag-us')
  assert.equal(getLanguageToggleLabel('en'), 'English')
  assert.equal(getLanguageToggleTitle('en'), 'Switch to 中文')
})

test('maps Chinese to the China flag class and label', () => {
  assert.equal(getLanguageFlagClass('zh'), 'language-flag-cn')
  assert.equal(getLanguageToggleLabel('zh'), '中文')
  assert.equal(getLanguageToggleTitle('zh'), 'Switch to English')
})

test('falls back to English flag content for unsupported languages', () => {
  assert.equal(getLanguageFlagClass('fr'), 'language-flag-us')
  assert.equal(getLanguageToggleLabel('fr'), 'English')
})
