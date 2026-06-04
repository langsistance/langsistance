import test from 'node:test'
import assert from 'node:assert/strict'

import { orderDownloadArtifacts } from './downloadArtifacts.js'

test('orderDownloadArtifacts puts Excel before CSV regardless of backend order', () => {
  const csv = { artifactId: 'csv-1', format: 'csv', complete: true }
  const xlsx = { artifactId: 'xlsx-1', format: 'xlsx', complete: true }

  assert.deepEqual(
    orderDownloadArtifacts([csv, xlsx]).map((artifact) => artifact.format),
    ['xlsx', 'csv']
  )
})

test('orderDownloadArtifacts keeps other complete artifacts after known formats', () => {
  const json = { artifactId: 'json-1', format: 'json', complete: true }
  const csv = { artifactId: 'csv-1', format: 'csv', complete: true }
  const xlsx = { artifactId: 'xlsx-1', format: 'xlsx', complete: true }

  assert.deepEqual(
    orderDownloadArtifacts([json, csv, xlsx]).map((artifact) => artifact.format),
    ['xlsx', 'csv', 'json']
  )
})

test('orderDownloadArtifacts omits incomplete artifacts', () => {
  const csv = { artifactId: 'csv-1', format: 'csv', complete: false }
  const xlsx = { artifactId: 'xlsx-1', format: 'xlsx', complete: true }

  assert.deepEqual(
    orderDownloadArtifacts([csv, xlsx]).map((artifact) => artifact.format),
    ['xlsx']
  )
})
