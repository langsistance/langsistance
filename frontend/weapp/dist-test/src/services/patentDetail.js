"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.specTarget = specTarget;
exports.claimsTarget = claimsTarget;
exports.normalizeSpec = normalizeSpec;
exports.normalizeClaims = normalizeClaims;
exports.fetchSpec = fetchSpec;
exports.fetchClaims = fetchClaims;
const api_1 = require("./api");
/** spec 用 patentId 优先，回退 applicationNumber（对齐 web SpecTab.tsx:19）。 */
function specTarget(patentId, applicationNumber) {
    return patentId || applicationNumber || '';
}
/** claims 用 applicationNumber 优先，回退 patentId（对齐 web ClaimsTab.tsx:21）。 */
function claimsTarget(patentId, applicationNumber) {
    return applicationNumber || patentId || '';
}
function normalizeSpec(body) {
    const message = String((body && body.message) || '');
    if (!body || body.success !== true) {
        return { ok: false, pdfUrl: '', message: message || '未找到说明书' };
    }
    const pdfUrl = String(body.pdf_url || '');
    if (!pdfUrl) {
        // success 但没给链接：当作失败，不要把空串交给下载层
        return { ok: false, pdfUrl: '', message: message || '未找到说明书' };
    }
    return { ok: true, pdfUrl, message };
}
function normalizeClaims(body) {
    const message = String((body && body.message) || '');
    const pdfUrl = String((body && body.pdf_url) || '');
    if (!body || body.success !== true) {
        return { ok: false, claims: [], pdfUrl: '', message: message || '未找到权利要求' };
    }
    const raw = Array.isArray(body.claims) ? body.claims : [];
    // 只留渲染要用的三个字段。status 的取值语义未核实，不猜、不透传。
    const claims = raw.map((c) => ({
        number: Number(c && c.number) || 0,
        text: String((c && c.text) || ''),
        independent: Boolean(c && c.independent),
    }));
    if (claims.length === 0 && !pdfUrl) {
        return { ok: false, claims: [], pdfUrl: '', message: message || '未找到权利要求' };
    }
    return { ok: true, claims, pdfUrl, message };
}
async function fetchSpec(source, patentId, applicationNumber = '') {
    const target = specTarget(patentId, applicationNumber);
    if (!target)
        return { ok: false, pdfUrl: '', message: '该条缺少可查询的专利号' };
    const body = await (0, api_1.request)(`/patent/${encodeURIComponent(source)}/${encodeURIComponent(target)}/spec`);
    return normalizeSpec(body);
}
async function fetchClaims(source, patentId, applicationNumber = '') {
    const target = claimsTarget(patentId, applicationNumber);
    if (!target)
        return { ok: false, claims: [], pdfUrl: '', message: '该条缺少可查询的专利号' };
    const body = await (0, api_1.request)(`/patent/${encodeURIComponent(source)}/${encodeURIComponent(target)}/claims`);
    return normalizeClaims(body);
}
//# sourceMappingURL=patentDetail.js.map