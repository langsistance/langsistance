import test from 'node:test'
import assert from 'node:assert/strict'

import { assertApiResponseSuccess } from './apiResponse.js'

test('throws backend message when API envelope reports failure', () => {
  assert.throws(
    () => assertApiResponseSuccess({ success: false, message: 'Knowledge update rejected' }, 'Update failed'),
    /Knowledge update rejected/
  )
})

test('falls back when API failure envelope has no message', () => {
  assert.throws(
    () => assertApiResponseSuccess({ success: false }, 'Update failed'),
    /Update failed/
  )
})

test('returns successful API envelopes unchanged', () => {
  const result = { success: true, data: { id: 1 } }

  assert.equal(assertApiResponseSuccess(result, 'Update failed'), result)
})
