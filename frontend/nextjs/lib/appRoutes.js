/**
 * App-router route helpers.
 *
 * next.config sets `trailingSlash: true` with a static export
 * (`output: 'export'`).  In that mode the client-side flight fetch appends
 * `index.txt` only when the target pathname ends with '/'; a programmatic
 * router.push with a bare pathname instead requests `<pathname>.txt`
 * (which does not exist) and Next silently falls back to a full-page
 * navigation — resetting every in-memory store (ChatProvider messages
 * included).  `<Link>` hrefs are normalized for us, but programmatic
 * pushes are not, so every one of them must carry the trailing slash.
 */
export function resultsPath(setId, sessionId) {
  const params = new URLSearchParams({ set: setId })
  if (sessionId) params.set('session_id', sessionId)
  return `/app/results/?${params.toString()}`
}

export function chatPath(query = '') {
  return `/app/chat/${query ? `?${query}` : ''}`
}
