// Temporary verification script: renders /app/chat with a fake auth token,
// screenshots the sidebar in expanded + collapsed states, and dumps computed
// styles to diagnose the collapsed-rail rendering.
import puppeteer from 'puppeteer-core'

const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
const URL = 'http://localhost:3000/app/chat'

const fakeAuth = {
  idToken: 'fake-id-token',
  refreshToken: 'fake-refresh-token',
  expiresAt: Date.now() + 24 * 3600 * 1000,
  uid: 'verify-user',
  email: 'verify@test.local',
}

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: 'new',
  args: ['--no-sandbox', '--disable-gpu'],
})

try {
  const page = await browser.newPage()
  await page.setViewport({ width: 1440, height: 900 })
  page.on('console', (m) => {
    if (m.type() === 'error') console.log('[console.error]', m.text().slice(0, 200))
  })
  page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 200)))

  // 1) Visit origin, inject fake auth, reload
  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: 30000 })
  await page.evaluate((auth) => {
    localStorage.setItem('cp_auth_v1', JSON.stringify(auth))
    localStorage.removeItem('sidebarCollapsed')
  }, fakeAuth)
  await page.reload({ waitUntil: 'domcontentloaded', timeout: 45000 })
  await page.waitForSelector('.sidebar', { timeout: 15000 })

  // Wait out the dev-server hydration / API calls
  await new Promise((r) => setTimeout(r, 3000))

  const dump = () =>
    page.evaluate(() => {
      const side = document.querySelector('.sidebar')
      const btns = [...document.querySelectorAll('.nav-item')]
      const visibleIcons = btns.filter((b) => {
        const r = b.getBoundingClientRect()
        return r.width > 0 && r.height > 0
      }).map((b) => ({
        w: Math.round(b.getBoundingClientRect().width),
        labelVisible: [...b.querySelectorAll('span')].some((s) => {
          const st = getComputedStyle(s)
          return st.display !== 'none' && st.visibility !== 'hidden'
        }),
      }))
      const collapseBtn = document.querySelector('.sidebar-collapse-btn')
      const cbr = collapseBtn?.getBoundingClientRect()
      return {
        sidebarClass: side?.className,
        sidebarWidth: side ? Math.round(side.getBoundingClientRect().width) : null,
        navItemCount: btns.length,
        visibleIcons,
        collapseBtnRect: cbr ? { x: Math.round(cbr.x), y: Math.round(cbr.y), w: Math.round(cbr.width), h: Math.round(cbr.height) } : null,
        sessionHistoryVisible: (() => {
          const el = document.querySelector('.session-history')
          return el ? getComputedStyle(el).display : 'no-el'
        })(),
        devSwitchVisible: (() => {
          const el = document.querySelector('.switch-wrap')
          return el ? getComputedStyle(el).display !== 'none' : 'no-el'
        })(),
      }
    })

  console.log('=== EXPANDED ===')
  console.log(JSON.stringify(await dump(), null, 2))
  await page.screenshot({ path: 'scripts/sidebar-expanded.png' })

  // 2) Collapse
  await page.click('.sidebar-collapse-btn')
  await new Promise((r) => setTimeout(r, 600)) // width transition 0.2s
  console.log('=== COLLAPSED ===')
  console.log(JSON.stringify(await dump(), null, 2))
  await page.screenshot({ path: 'scripts/sidebar-collapsed.png' })

  // 3) Expand again (restore button must be reachable)
  await page.click('.sidebar-collapse-btn')
  await new Promise((r) => setTimeout(r, 600))
  console.log('=== RE-EXPANDED width ===', (await dump()).sidebarWidth)
} finally {
  await browser.close()
}
