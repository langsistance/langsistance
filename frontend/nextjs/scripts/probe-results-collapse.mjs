// Probe: does the results page render the chat-collapse button, and does it work?
// Usage: node scripts/probe-results-collapse.mjs [url]
import puppeteer from 'puppeteer-core'

const url = process.argv[2] || 'http://localhost:3000/app/results?set=probe1'
const edgePath = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'

const browser = await puppeteer.launch({
  executablePath: edgePath,
  headless: 'new',
  args: ['--no-sandbox'],
})
const page = await browser.newPage()

// Inject fake auth + results store BEFORE any page script runs
await page.evaluateOnNewDocument(() => {
  const auth = {
    uid: 'probe-uid',
    email: 'probe@test.dev',
    idToken: 'fake-token',
    refreshToken: 'fake-refresh',
    expiresAt: Date.now() + 3600_000,
  }
  const store = {
    sets: {
      probe1: {
        setId: 'probe1',
        source: 'uspto',
        columns: [{ key: 'patent_id', label: '专利号', role: 'text' }],
        rows: [{ patent_id: '12096133' }, { patent_id: '16622181' }],
      },
    },
    index: [{ setId: 'probe1', sessionId: null, queryText: 'probe query', savedAt: Date.now() }],
  }
  localStorage.setItem('cp_auth_v1', JSON.stringify(auth))
  localStorage.setItem('copiioai_results', JSON.stringify(store))
})

page.on('console', (m) => {
  const txt = m.text()
  if (/error|uncaught|failed/i.test(txt)) console.log('[console]', txt.slice(0, 300))
})
page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 300)))

await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 }).catch((e) => console.log('[goto]', String(e).slice(0, 200)))
await new Promise((r) => setTimeout(r, 2000))

const report = await page.evaluate(() => {
  const out = {
    url: location.href,
    hasLayout: !!document.querySelector('.results-layout'),
    hasSidebar: !!document.querySelector('.results-chat-sidebar'),
    hasHeader: !!document.querySelector('.results-chat-header'),
    hasCollapseBtn: !!document.querySelector('.results-collapse-btn'),
    hasExpandBtn: !!document.querySelector('.results-expand-btn'),
    hasEmpty: !!document.querySelector('.results-empty'),
    bodyText: document.body.innerText.slice(0, 200).replace(/\n+/g, ' | '),
  }
  const btn = document.querySelector('.results-collapse-btn')
  if (btn) {
    const r = btn.getBoundingClientRect()
    const cs = getComputedStyle(btn)
    out.btnRect = { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height), display: cs.display, visibility: cs.visibility, opacity: cs.opacity }
  }
  const hdr = document.querySelector('.results-chat-header')
  if (hdr) {
    const r = hdr.getBoundingClientRect()
    out.headerRect = { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) }
  }
  const sidebar = document.querySelector('.results-chat-sidebar')
  if (sidebar) {
    const r = sidebar.getBoundingClientRect()
    out.sidebarRect = { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) }
  }
  const layout = document.querySelector('.results-layout')
  if (layout) {
    out.layoutClass = layout.className
    out.layoutGrid = getComputedStyle(layout).gridTemplateColumns
  }
  return out
})
console.log(JSON.stringify(report, null, 2))

// If the collapse button exists, click it and see whether the layout changes
if (report.hasCollapseBtn) {
  await page.click('.results-collapse-btn')
  await new Promise((r) => setTimeout(r, 500))
  const after = await page.evaluate(() => {
    const layout = document.querySelector('.results-layout')
    return {
      layoutClass: layout?.className,
      layoutGrid: layout ? getComputedStyle(layout).gridTemplateColumns : null,
      hasSidebar: !!document.querySelector('.results-chat-sidebar'),
      hasExpandBtn: !!document.querySelector('.results-expand-btn'),
    }
  })
  console.log('AFTER CLICK:', JSON.stringify(after, null, 2))
}

await browser.close()
