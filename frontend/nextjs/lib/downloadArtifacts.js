const DOWNLOAD_FORMAT_ORDER = ['xlsx', 'csv']
const VISUAL_LABELS = {
  xlsx: 'XLSX',
  csv: 'CSV',
}

export function artifactVisualLabel(format) {
  return VISUAL_LABELS[format] ?? String(format || '').toUpperCase()
}

export function orderDownloadArtifacts(artifacts) {
  const completeArtifacts = artifacts.filter((artifact) => artifact?.complete)
  const known = DOWNLOAD_FORMAT_ORDER
    .map((format) => completeArtifacts.find((artifact) => artifact.format === format))
    .filter(Boolean)
  const knownFormats = new Set(DOWNLOAD_FORMAT_ORDER)
  const other = completeArtifacts.filter((artifact) => !knownFormats.has(artifact.format))

  return [...known, ...other]
}
