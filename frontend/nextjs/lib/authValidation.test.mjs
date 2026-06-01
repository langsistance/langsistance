import test from 'node:test'
import assert from 'node:assert/strict'

import { validateSignupPasswordConfirmation } from './authValidation.js'

test('signup validation rejects mismatched password confirmation in English', () => {
  assert.equal(
    validateSignupPasswordConfirmation('secret123', 'secret456', 'en'),
    'Passwords do not match'
  )
})

test('signup validation rejects mismatched password confirmation in Chinese', () => {
  assert.equal(
    validateSignupPasswordConfirmation('secret123', 'secret456', 'zh'),
    '两次输入的密码不一致'
  )
})

test('signup validation accepts matching password confirmation', () => {
  assert.equal(validateSignupPasswordConfirmation('secret123', 'secret123', 'en'), '')
})
