export function buildCustomToolImportPayload(spec) {
  if (!spec || typeof spec !== 'object' || Array.isArray(spec)) {
    throw new TypeError('Custom tool spec must be a JSON object');
  }

  return {
    spec_format: 'json',
    spec_content: JSON.stringify(spec),
  };
}
