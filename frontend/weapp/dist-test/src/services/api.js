"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.clearAuthAndRedirect = clearAuthAndRedirect;
exports.request = request;
exports.errorText = errorText;
const taro_1 = require("@tarojs/taro");
const config_1 = require("../config");
/**
 * 凭证失效的统一处置：清本地 + 回登录页 + 抛出可见错误。
 * 抽出来是因为 Taro.uploadFile / Taro.downloadFile 走不到 request()，
 * 但同样需要这套处置。
 */
function clearAuthAndRedirect(message = '登录已失效') {
    taro_1.default.removeStorageSync(config_1.STORAGE_KEYS.wxToken);
    taro_1.default.removeStorageSync(config_1.STORAGE_KEYS.userId);
    taro_1.default.navigateTo({ url: '/pages/login/index' });
    throw new Error(message);
}
async function request(path, options = {}) {
    const { method = 'GET', data, auth = true } = options;
    const header = {
        'content-type': 'application/json',
    };
    if (auth) {
        const token = taro_1.default.getStorageSync(config_1.STORAGE_KEYS.wxToken);
        if (token) {
            header.Authorization = `Bearer ${token}`;
        }
    }
    const resp = await taro_1.default.request({
        url: `${config_1.API_BASE}${path}`,
        method,
        data,
        header,
        timeout: 30000,
    });
    const body = resp.data;
    // 只有**带鉴权**的请求才把 401 当会话失效。
    //
    // 登录/注册这类 auth:false 的请求没有会话可失效；`/auth/wechat` 兑换 code
    // 失败时回的正是 401（api_routes/wechat_auth.py:53）。若也走这里，就会：
    // 清空凭证 → navigateTo 到登录页（而我们**已经在登录页上**）→ 抛出笼统的
    // "登录已失效"，把后端的真实 detail（如 "code2session error 40029: ..."）
    // 盖掉。用户于是只看到一个转圈和一个无信息量的错误，真正的原因消失。
    if (auth && resp.statusCode === 401) {
        clearAuthAndRedirect((body && (body.detail || body.message)) || '登录已失效');
    }
    if (resp.statusCode >= 400) {
        const detail = body && (body.detail || body.message);
        throw new Error(typeof detail === 'string' ? detail : `请求失败(${resp.statusCode})`);
    }
    return body;
}
/**
 * 通用取错误文案。
 *
 * 三条来源，缺一不可：
 *  1. 后端 HTTP 错误体：`detail`（FastAPI HTTPException）/ `message`
 *  2. JS `Error`：`message`
 *  3. **微信网络层 fail 对象：`errMsg`** —— Taro 的 promise 包装在
 *     `wx.request` 失败时 reject 的是这个，其上是 `errMsg`
 *     （如 `request:fail url not in domain list`）。
 *
 * 第 3 条曾漏掉，后果很严重：域名不在白名单、证书不合法、断网等**根本没到
 * 后端**的失败全被吞成兜底文案，用户只看到"请稍后重试"，而服务端**一条日志
 * 都没有**——排查时极难定位。upload.ts 早就单独处理了 errMsg，这里漏了。
 */
function errorText(err, fallback = '操作失败，请稍后重试') {
    const detail = err && (err.detail || err.message);
    if (typeof detail === 'string')
        return detail;
    if (detail && typeof detail.error === 'string')
        return detail.error;
    if (err && typeof err.message === 'string')
        return err.message;
    if (err && typeof err.errMsg === 'string')
        return err.errMsg;
    return fallback;
}
//# sourceMappingURL=api.js.map