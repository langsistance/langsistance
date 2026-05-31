function stringifyErrorValue(value) {
  if (typeof value === 'string') return value.trim()
  if (value == null) return ''
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

export function getApiResponseErrorMessage(result, fallbackMessage = 'Request failed') {
  if (!result || typeof result !== 'object' || result.success !== false) {
    return ''
  }

  return (
    stringifyErrorValue(result.message) ||
    stringifyErrorValue(result.error) ||
    stringifyErrorValue(result.detail) ||
    fallbackMessage
  )
}

export function assertApiResponseSuccess(result, fallbackMessage = 'Request failed') {
  const message = getApiResponseErrorMessage(result, fallbackMessage)
  if (message) {
    throw new Error(message)
  }

  return result
}
