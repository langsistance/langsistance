import test from 'node:test'
import assert from 'node:assert/strict'

import { copyTextToClipboard } from './clipboard.js'

function createDocument() {
  const appended = []
  const removed = []
  const commands = []
  const selected = []

  return {
    appended,
    removed,
    commands,
    createElement(tagName) {
      assert.equal(tagName, 'textarea')
      return {
        value: '',
        style: {},
        setAttribute(name, value) {
          this[name] = value
        },
        select() {
          selected.push(this.value)
        },
      }
    },
    body: {
      appendChild(element) {
        appended.push(element)
      },
      removeChild(element) {
        removed.push(element)
      },
    },
    execCommand(command) {
      commands.push(command)
      return true
    },
    selected,
  }
}

test('copyTextToClipboard uses navigator clipboard when available', async () => {
  const calls = []

  const ok = await copyTextToClipboard('hello', {
    navigatorRef: {
      clipboard: {
        async writeText(text) {
          calls.push(text)
        },
      },
    },
    documentRef: createDocument(),
  })

  assert.equal(ok, true)
  assert.deepEqual(calls, ['hello'])
})

test('copyTextToClipboard falls back to execCommand when clipboard API is unavailable', async () => {
  const documentRef = createDocument()

  const ok = await copyTextToClipboard('hello fallback', {
    navigatorRef: {},
    documentRef,
  })

  assert.equal(ok, true)
  assert.equal(documentRef.appended.length, 1)
  assert.equal(documentRef.removed.length, 1)
  assert.deepEqual(documentRef.commands, ['copy'])
  assert.deepEqual(documentRef.selected, ['hello fallback'])
})

test('copyTextToClipboard falls back when clipboard write fails', async () => {
  const documentRef = createDocument()

  const ok = await copyTextToClipboard('retry fallback', {
    navigatorRef: {
      clipboard: {
        async writeText() {
          throw new Error('not allowed')
        },
      },
    },
    documentRef,
  })

  assert.equal(ok, true)
  assert.deepEqual(documentRef.commands, ['copy'])
  assert.deepEqual(documentRef.selected, ['retry fallback'])
})
