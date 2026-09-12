import test from 'node:test';
import assert from 'node:assert/strict';
// 必须排在最前：补上 Taro 构建期才会 inlined 的裸标识符常量，
// 否则下一行 import api → @tarojs/taro → @tarojs/runtime 会直接 ReferenceError。
import './taroRuntimeStubs.js';
import TaroModule from '@tarojs/taro';
import { request, errorText } from './api.js';
/** CJS/ESM 互操作下默认导出可能是外层命名空间，两种形态都取一下。 */
const Taro = TaroModule?.default?.request ? TaroModule.default : TaroModule;
/**
 * 把 Taro 的 IO 面换成桩，返回调用账本。
 *
 * `api.js` 编译成 CommonJS 后是 `taro_1.default.request(...)`，所以补在这里
 * 的桩会被它看到。
 */
function stubTaro({ statusCode, data }) {
    const ledger = { requested: [], navigated: [], removedKeys: [] };
    const saved = {
        request: Taro.request,
        navigateTo: Taro.navigateTo,
        removeStorageSync: Taro.removeStorageSync,
        getStorageSync: Taro.getStorageSync,
    };
    Taro.request = async (opts) => {
        ledger.requested.push(opts);
        return { statusCode, data };
    };
    Taro.navigateTo = (opts) => {
        ledger.navigated.push(opts.url);
        return Promise.resolve();
    };
    Taro.removeStorageSync = (k) => void ledger.removedKeys.push(k);
    Taro.getStorageSync = () => 'stale-token';
    return { ledger, restore: () => Object.assign(Taro, saved) };
}
/** 后端 /auth/wechat 兑换 code 失败时的真实响应（api_routes/wechat_auth.py:39-53）。 */
const WECHAT_401 = {
    statusCode: 401,
    data: { detail: 'code2session error 40029: invalid code, rid: 6aa49951-6bff1dc9-11ab6f23' },
};
/**
 * 承重用例：登录请求（auth:false）拿到 401 时，必须把后端的**真实原因**抛出来，
 * 且**不能**触发"会话失效"处置。
 *
 * 修复前：`resp.statusCode === 401` 无条件走 clearAuthAndRedirect —— 清空凭证、
 * navigateTo 到登录页（而调用方**正在登录页上**），并把 detail 换成笼统的
 * "登录已失效"。用户只看到转圈 + 无信息量的错误，真因被盖掉。
 *
 * 判别力：断言的是 detail 文本与 navigateTo 的**调用次数**，两者修复前都不同 ——
 * 只看是否 reject 的实现无法通过（两条路径都 reject）。
 */
test('request：auth:false 的 401 透出后端真实 detail，且不跳登录页', async () => {
    const { ledger, restore } = stubTaro(WECHAT_401);
    try {
        await assert.rejects(() => request('/auth/wechat', { method: 'POST', data: { code: 'x' }, auth: false }), (err) => {
            assert.match(err.message, /40029/, '真实 detail 被吞掉了——登录失败的根因无法被用户看到');
            assert.doesNotMatch(err.message, /登录已失效/, '登录请求没有会话可失效，不该报这句');
            return true;
        });
        assert.deepEqual(ledger.navigated, [], 'auth:false 的 401 不该跳登录页');
        assert.deepEqual(ledger.removedKeys, [], 'auth:false 的 401 不该清凭证');
    }
    finally {
        restore();
    }
});
/**
 * 对照组：**带鉴权**的请求拿到 401，仍必须走会话失效处置。
 * 修复不能把这条真实路径一起关掉。
 */
test('request：auth:true 的 401 仍清凭证并跳登录页', async () => {
    const { ledger, restore } = stubTaro({ statusCode: 401, data: { detail: 'token expired' } });
    try {
        await assert.rejects(() => request('/sessions'), (err) => {
            assert.match(err.message, /token expired/);
            return true;
        });
        assert.deepEqual(ledger.navigated, ['/pages/login/index'], 'authenticated 401 必须回登录页');
        assert.ok(ledger.removedKeys.length > 0, 'authenticated 401 必须清掉本地凭证');
    }
    finally {
        restore();
    }
});
/**
 * 承重用例：微信**网络层**失败时 Taro reject 的是原始 fail 对象，其上只有
 * `errMsg`（没有 detail / message）。
 *
 * 修复前这个字段不被识别 → 一路落到兜底文案。后果在 2026-09-12 真机实测中
 * 暴露：体验版因域名未备案被拦（600002），用户只看到"登录失败，请稍后重试"，
 * 而**服务端一条日志都没有** —— 排查时完全无从下手。
 *
 * 判别力：断言真实 errMsg 文本出现。只看"是否返回兜底"的弱实现能过，但它正是漏掉的那条。
 */
test('errorText：微信 fail 对象的 errMsg 必须透出，不能被兜底吞掉', () => {
    const fail = { errno: 600002, errMsg: 'request:fail url not in domain list' };
    assert.equal(errorText(fail, '登录失败，请稍后重试'), 'request:fail url not in domain list');
});
test('errorText：后端 detail 优先于 errMsg', () => {
    assert.equal(errorText({ detail: 'code2session error 40029', errMsg: 'request:fail' }), 'code2session error 40029');
});
test('errorText：什么都没有时才用兜底', () => {
    assert.equal(errorText({}, '兜底'), '兜底');
    assert.equal(errorText(null, '兜底'), '兜底');
});
/** 非 401 的业务失败照常抛出，不受 401 分支影响。 */
test('request：auth:false 的 400 抛出 detail', async () => {
    const { restore } = stubTaro({ statusCode: 400, data: { detail: 'code is required' } });
    try {
        await assert.rejects(() => request('/auth/wechat', { method: 'POST', data: {}, auth: false }), /code is required/);
    }
    finally {
        restore();
    }
});
//# sourceMappingURL=api.test.mjs.map