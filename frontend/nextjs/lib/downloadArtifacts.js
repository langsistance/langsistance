const DOWNLOAD_FORMAT_ORDER = ['xlsx', 'csv']

export function orderDownloadArtifacts(artifacts) {
  const completeArtifacts = artifacts.filter((artifact) => artifact?.complete)
  const known = DOWNLOAD_FORMAT_ORDER
    .map((format) => completeArtifacts.find((artifact) => artifact.format === format))
    .filter(Boolean)
  const knownFormats = new Set(DOWNLOAD_FORMAT_ORDER)
  const other = completeArtifacts.filter((artifact) => !knownFormats.has(artifact.format))

  return [...known, ...other]
}
