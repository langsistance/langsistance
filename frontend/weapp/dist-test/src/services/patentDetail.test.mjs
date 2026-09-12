import test from 'node:test';
import assert from 'node:assert/strict';
// 必须排在最前：补上 Taro 构建期才会 inlined 的裸标识符常量，
// 否则下一行 import patentDetail → api → @tarojs/taro → @tarojs/runtime 会直接 ReferenceError。
import './taroRuntimeStubs.js';
import { normalizeSpec, normalizeClaims, specTarget, claimsTarget } from './patentDetail.js';
test('normalizeSpec：success:false 判为失败——HTTP 200 也可能是业务失败', () => {
    const out = normalizeSpec({ success: false, message: '未找到说明书' });
    assert.equal(out.ok, false);
    assert.equal(out.message, '未找到说明书');
    assert.equal(out.pdfUrl, '');
});
/**
 * 上面那条**不足以**把「看 success」与「看状态码」分开：它传的 body 没有 pdf_url，
 * 于是一个 200 即视为成功、只按「有没有 pdf_url」决定的实现会走到同一个 return，
 * 输出逐字节相同，照样绿——而它正是本任务要防的那个缺陷。
 *
 * 真正的业务失败（PatentDetailError 分支，patent_detail.py:774-778）返回的是
 * `{success:false, message:...}` 且**不带 pdf_url**；靠二者缺一不可才判失败，
 * 「success 为假」这一步是多余的。要逼出差异只有一种输入：pdf_url **在**
 * （所以状态码实现在此会判成功），而 success 为 **false**。
 * 说明书不可用却把链接交给下载层，正是要避免的后果。
 */
test('normalizeSpec：有 pdf_url 但 success:false 仍判失败——只看状态码的实现会在此放行', () => {
    const out = normalizeSpec({ success: false, pdf_url: 'https://x/stale.pdf' });
    assert.equal(out.ok, false, 'success:false 却放行了 pdf_url——判定看的是状态码/链接而非 success 字段');
    assert.equal(out.pdfUrl, '', '业务失败时不应把链接交给下载层');
});
test('normalizeSpec：success:true 取 pdf_url', () => {
    const out = normalizeSpec({ success: true, pdf_url: 'https://x/a.pdf' });
    assert.equal(out.ok, true);
    assert.equal(out.pdfUrl, 'https://x/a.pdf');
});
test('normalizeSpec：success 为真但无 pdf_url，判为失败而不是给空链接', () => {
    const out = normalizeSpec({ success: true });
    assert.equal(out.ok, false);
});
test('normalizeClaims：结构化权利要求归一，只留用到的三个字段', () => {
    // status 用后端真实的取值（patent_detail.py:130 active/canceled）。
    // 这里刻意用 'canceled'：若归一把 status 透传，它会原样出现在结果里——
    // 用假值 'x' 也能过，但那是靠"假值恰好不等于期望值"，测不出真实契约。
    const out = normalizeClaims({
        success: true,
        claims: [
            { number: 1, text: '一种…', status: 'canceled', independent: true },
            { number: 2, text: '根据权利要求1…', independent: false },
        ],
    });
    assert.equal(out.ok, true);
    assert.deepEqual(out.claims, [
        { number: 1, text: '一种…', independent: true },
        { number: 2, text: '根据权利要求1…', independent: false },
    ]);
    // 显式一点：status 的语义未核实，不能透传给渲染层
    assert.ok(!('status' in out.claims[0]), 'status 被透传了');
});
test('normalizeClaims：无结构化但有 pdf_url → ok 且 claims 为空，调用方回退 PDF', () => {
    const out = normalizeClaims({ success: true, pdf_url: 'https://x/c.pdf' });
    assert.equal(out.ok, true);
    assert.equal(out.claims.length, 0);
    assert.equal(out.pdfUrl, 'https://x/c.pdf');
});
test('normalizeClaims：既无 claims 也无 pdf_url → 失败', () => {
    assert.equal(normalizeClaims({ success: true }).ok, false);
});
/**
 * 与 normalizeSpec 同理：业务失败（PatentDetailError 分支）返回 `{success:false, message}`，
 * 不带 claims。上面那两条测试传的 body 恰好都是「空 claims + 无 pdf_url」，
 * 「success 为假」这一步在它们眼里是多余的——只看状态码的实现照样绿。
 * 这里把 claims **放进来**（状态码实现会判成功、把失败响应的 claims 渲染出来），
 * 而 success 为 **false**。
 */
test('normalizeClaims：有 claims 但 success:false 仍判失败——只看状态码的实现会在此放行', () => {
    const out = normalizeClaims({
        success: false,
        claims: [{ number: 1, text: '一种…', independent: true }],
    });
    assert.equal(out.ok, false, 'success:false 却放行了 claims——判定看的是状态码而非 success 字段');
    assert.equal(out.claims.length, 0);
});
test('spec 用 patentId 优先，claims 用 applicationNumber 优先（对齐 web）', () => {
    assert.equal(specTarget('US1', '17638216'), 'US1');
    assert.equal(specTarget('', '17638216'), '17638216');
    assert.equal(specTarget('', ''), '');
    assert.equal(claimsTarget('US1', '17638216'), '17638216');
    assert.equal(claimsTarget('US1', ''), 'US1');
    assert.equal(claimsTarget('', ''), '');
});
//# sourceMappingURL=patentDetail.test.mjs.map