import puppeteer from 'puppeteer-core'

const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe'

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--no-sandbox', '--disable-gpu'],
})

const page = await browser.newPage()
page.on('console', (m) => {
  const t = m.text()
  if (t.length > 0 && !t.includes('Manifest') && !t.includes('Download the React DevTools')) {
    console.log('[console]', t.slice(0, 320))
  }
})
page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 300)))
const fullLoads = []
page.on('framenavigated', (f) => {
  if (f === page.mainFrame()) {
    fullLoads.push(f.url())
    console.log('[full-load]', f.url())
  }
})
const cdp = await page.createCDPSession()
await cdp.send('Network.enable')
await cdp.send('Page.enable')
cdp.on('Page.frameRequestedNavigation', (e) => {
  console.log('[cdp-nav-request]', JSON.stringify({ url: (e.url || '').slice(0, 120), reason: e.reason, disposition: e.disposition }))
})
cdp.on('Page.frameNavigated', (e) => {
  if (e.frame?.parentId) return
  console.log('[cdp-frame-navigated]', JSON.stringify({ url: (e.frame?.url || '').slice(0, 120), type: e.type }))
})
cdp.on('Network.requestWillBeSent', (e) => {
  if (e.type === 'Document' || e.request.url.includes('results') || e.request.url.includes('chat')) {
    console.log('[cdp-doc-req]', e.request.url.slice(0, 130))
    if (e.initiator && e.initiator.stack) {
      console.log('[cdp-initiator]', JSON.stringify(e.initiator).slice(0, 600))
    }
  }
})

console.log('=== goto /app/chat/ (static, localhost:8123)')
await page.goto('http://localhost:8123/app/chat/', { waitUntil: 'domcontentloaded', timeout: 30000 })
await new Promise((r) => setTimeout(r, 3000))
console.log('=== landed:', page.url())
console.log(
  '=== state:',
  await page.evaluate(() => ({
    routerExposed: Boolean(window.__diagRouter),
    navItems: document.querySelectorAll('a').length,
  })),
)

const pushed = await page.evaluate(() => {
  for (const m of ['assign', 'replace', 'reload']) {
    const orig = window.location[m].bind(window.location)
    window.location[m] = (...args) => {
      console.log('[loc-diag] location.' + m, JSON.stringify(args))
      return orig(...args)
    }
  }
  const ps = history.pushState.bind(history)
  history.pushState = (...args) => {
    console.log('[loc-diag] pushState', JSON.stringify(args[2]))
    return ps(...args)
  }
  const rs = history.replaceState.bind(history)
  history.replaceState = (...args) => {
    console.log('[loc-diag] replaceState', JSON.stringify(args[2]))
    return rs(...args)
  }
  for (const m of ['go', 'back', 'forward']) {
    const orig = history[m].bind(history)
    history[m] = (...args) => {
      console.log('[loc-diag] history.' + m, JSON.stringify(args))
      return orig(...args)
    }
  }
  const hd = Object.getOwnPropertyDescriptor(window.location, 'href')
  Object.defineProperty(window.Location.prototype, 'href', {
    get: hd.get,
    set: function (v) {
      console.log('[loc-diag] href setter ->', v, '| stack:', String(new Error().stack).split(String.fromCharCode(10)).slice(2, 5).join(' | '))
      return hd.set.call(this, v)
    },
    configurable: true,
  })
  try {
    window.__diagRouter.push('/app/results/?set=diagtest')
    return true
  } catch (e) {
    return String(e)
  }
})
console.log('=== router.push called:', pushed)
await new Promise((r) => setTimeout(r, 8000))
console.log('=== final url:', page.url())
console.log('=== full page loads after goto:', JSON.stringify(fullLoads.slice(1)))
await browser.close()
