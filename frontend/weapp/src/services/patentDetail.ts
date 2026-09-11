import { request } from './api'

/**
 * 专利详情接口：说明书（spec）与权利要求（claims）。
 *
 * ⚠️ 这两条接口**失败时返回 HTTP 200 + {success:false}**，不是 5xx——
 * Cloudflare 会替换源站的 5xx 页面并导致 CORS 失败（api_routes/patent_detail.py:774-778）。
 * 所以判定必须看 `success` 字段，只看状态码会漏掉全部业务失败。
 *
 * 归一层是纯函数，故可测；网络部分只做参数拼装与转发。
 */

export interface ClaimItem {
  number: number
  text: string
  independent: boolean
}

export interface SpecResult {
  ok: boolean
  pdfUrl: string
  message: string
}

export interface ClaimsResult {
  ok: boolean
  claims: ClaimItem[]
  pdfUrl: string
  message: string
}

/** spec 用 patentId 优先，回退 applicationNumber（对齐 web SpecTab.tsx:19）。 */
export function specTarget(patentId: string, applicationNumber: string): string {
  return patentId || applicationNumber || ''
}

/** claims 用 applicationNumber 优先，回退 patentId（对齐 web ClaimsTab.tsx:21）。 */
export function claimsTarget(patentId: string, applicationNumber: string): string {
  return applicationNumber || patentId || ''
}

export function normalizeSpec(body: any): SpecResult {
  const message = String((body && body.message) || '')
  if (!body || body.success !== true) {
    return { ok: false, pdfUrl: '', message: message || '未找到说明书' }
  }
  const pdfUrl = String(body.pdf_url || '')
  if (!pdfUrl) {
    // success 但没给链接：当作失败，不要把空串交给下载层
    return { ok: false, pdfUrl: '', message: message || '未找到说明书' }
  }
  return { ok: true, pdfUrl, message }
}

export function normalizeClaims(body: any): ClaimsResult {
  const message = String((body && body.message) || '')
  const pdfUrl = String((body && body.pdf_url) || '')
  if (!body || body.success !== true) {
    return { ok: false, claims: [], pdfUrl: '', message: message || '未找到权利要求' }
  }
  const raw = Array.isArray(body.claims) ? body.claims : []
  // 只留渲染要用的三个字段。status 的取值语义未核实，不猜、不透传。
  const claims: ClaimItem[] = raw.map((c: any) => ({
    number: Number(c && c.number) || 0,
    text: String((c && c.text) || ''),
    independent: Boolean(c && c.independent),
  }))
  if (claims.length === 0 && !pdfUrl) {
    return { ok: false, claims: [], pdfUrl: '', message: message || '未找到权利要求' }
  }
  return { ok: true, claims, pdfUrl, message }
}

export async function fetchSpec(
  source: string,
  patentId: string,
  applicationNumber = '',
): Promise<SpecResult> {
  const target = specTarget(patentId, applicationNumber)
  if (!target) return { ok: false, pdfUrl: '', message: '该条缺少可查询的专利号' }
  const body = await request(
    `/patent/${encodeURIComponent(source)}/${encodeURIComponent(target)}/spec`,
  )
  return normalizeSpec(body)
}

export async function fetchClaims(
  source: string,
  patentId: string,
  applicationNumber = '',
): Promise<ClaimsResult> {
  const target = claimsTarget(patentId, applicationNumber)
  if (!target) return { ok: false, claims: [], pdfUrl: '', message: '该条缺少可查询的专利号' }
  const body = await request(
    `/patent/${encodeURIComponent(source)}/${encodeURIComponent(target)}/claims`,
  )
  return normalizeClaims(body)
}
