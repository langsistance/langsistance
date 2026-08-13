import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildRowModel,
  findRoleColumn,
  pruneResultsForPersistence,
  resolveActiveResultsMessage,
} from './results.js'

const COLUMNS = [
  { key: 'patentTitle', label: '标题', role: 'title' },
  { key: 'patentNumber', label: '专利号', role: 'patent_id' },
  { key: 'applicationNumberText', label: '申请号', role: 'application_number' },
  { key: 'applicationMetaData.earliestPublicationNumber', label: '公开号', role: 'publication_number' },
  { key: 'assigneeEntityName', label: '申请人', role: 'assignee' },
  { key: 'publicationDate', label: '公开日', role: 'publication_date' },
  { key: 'abstractText', label: '摘要', role: 'abstract' },
  { key: 'customThing', label: '自定义', role: 'text' },
]

const ROW = {
  patentTitle: '一种图像处理方法',
  patentNumber: 'US12000123B2',
  applicationNumberText: '17638216',
  'applicationMetaData.earliestPublicationNumber': 'US20220294065A1',
  assigneeEntityName: '华为',
  publicationDate: '2024-06-01',
  abstractText: '摘要文字。',
  customThing: 'x',
}

test('findRoleColumn returns the first column matching a role', () => {
  assert.deepEqual(findRoleColumn(COLUMNS, 'title'), COLUMNS[0])
  assert.deepEqual(findRoleColumn(COLUMNS, 'patent_id'), COLUMNS[1])
  assert.equal(findRoleColumn(COLUMNS, 'url'), null)
})

test('buildRowModel builds title/meta/patent identifiers for patent rows', () => {
  const model = buildRowModel(ROW, COLUMNS, 'uspto')
  assert.equal(model.title, '一种图像处理方法')
  assert.equal(model.patentId, 'US12000123B2')
  assert.equal(model.applicationNumber, '17638216')
  assert.equal(model.isDocument, false)
  assert.equal(model.meta.length, 5) // patent_id + application_number + publication_number + assignee + publication_date
  assert.equal(model.meta[0].label, '专利号')
  assert.ok(model.fields.length >= 6) // all non-empty fields incl. text role
})

test('buildRowModel detects document rows by role', () => {
  const docColumns = [
    { key: 'documentTitle', label: '文档标题', role: 'document_title' },
    { key: 'documentDate', label: '日期', role: 'document_date' },
    { key: 'pdfUrl', label: '链接', role: 'url' },
  ]
  const docRow = { documentTitle: 'Issue Notification', documentDate: '2025-09-24', pdfUrl: 'https://x/y.pdf' }
  const model = buildRowModel(docRow, docColumns, 'uspto_documents')
  assert.equal(model.isDocument, true)
  assert.equal(model.title, 'Issue Notification')
  assert.equal(model.url, 'https://x/y.pdf')
})

test('buildRowModel tolerates empty rows', () => {
  const model = buildRowModel({}, COLUMNS, 'uspto')
  assert.equal(model.title, '')
  assert.equal(model.patentId, '')
  assert.equal(model.fields.length, 0)
})

test('pruneResultsForPersistence truncates abstracts and caps rows', () => {
  const longAbstract = '字'.repeat(1200)
  const rows = Array.from({ length: 60 }, (_, i) => ({
    ...ROW,
    patentNumber: `US${i}`,
    abstractText: longAbstract,
  }))
  const pruned = pruneResultsForPersistence(
    { setId: 's1', source: 'uspto', columns: COLUMNS, rows },
  )
  assert.equal(pruned.rows.length, 50)
  assert.ok(pruned.rows[0].abstractText.length <= 500)
  assert.equal(pruned.rows[0].patentNumber, 'US0')
  assert.equal('customThing' in pruned.rows[0], false)
})

const RESULT_MESSAGE = (setId) => ({ id: `m-${setId}`, role: 'assistant', results: { setId, source: 'uspto', columns: [], rows: [] } })

test('resolveActiveResultsMessage returns the exact setId match', () => {
  const messages = [
    { id: 'u1', role: 'user', content: 'hi' },
    RESULT_MESSAGE('set-a'),
    RESULT_MESSAGE('set-b'),
  ]
  assert.equal(resolveActiveResultsMessage(messages, 'set-a').id, 'm-set-a')
})

test('resolveActiveResultsMessage falls back to newest results when setId misses', () => {
  const messages = [RESULT_MESSAGE('set-a'), RESULT_MESSAGE('set-b')]
  // Stale/absent setId (timing race after auto-navigation) — newest wins
  assert.equal(resolveActiveResultsMessage(messages, 'set-missing').id, 'm-set-b')
})

test('resolveActiveResultsMessage returns newest results when setId is null', () => {
  const messages = [
    { id: 'u1', role: 'user', content: 'hi' },
    RESULT_MESSAGE('set-a'),
    RESULT_MESSAGE('set-b'),
  ]
  assert.equal(resolveActiveResultsMessage(messages, null).id, 'm-set-b')
  assert.equal(resolveActiveResultsMessage(messages, '').id, 'm-set-b')
})

test('resolveActiveResultsMessage returns undefined with no results anywhere', () => {
  const messages = [{ id: 'u1', role: 'user', content: 'hi' }]
  assert.equal(resolveActiveResultsMessage(messages, 'set-a'), undefined)
  assert.equal(resolveActiveResultsMessage([], 'set-a'), undefined)
  assert.equal(resolveActiveResultsMessage(null, 'set-a'), undefined)
})

test('resolveActiveResultsMessage falls back to stored set before newest fallback', () => {
  const messages = [RESULT_MESSAGE('set-a')]
  const store = { sets: { 'stored-x': { source: 'uspto', columns: [], rows: [] } }, index: [] }
  const resolved = resolveActiveResultsMessage(messages, 'stored-x', store)
  assert.equal(resolved.id, 'stored-stored-x')
  assert.equal(resolved.results.setId, 'stored-x')
})

test('resolveActiveResultsMessage prefers exact memory match over store', () => {
  const messages = [RESULT_MESSAGE('set-a')]
  const store = { sets: { 'set-a': { source: 'uspto', columns: [], rows: [] } }, index: [] }
  assert.equal(resolveActiveResultsMessage(messages, 'set-a', store).id, 'm-set-a')
})

test('resolveActiveResultsMessage keeps newest fallback when store misses', () => {
  const messages = [RESULT_MESSAGE('set-a')]
  const store = { sets: {}, index: [] }
  assert.equal(resolveActiveResultsMessage(messages, 'missing', store).id, 'm-set-a')
})
