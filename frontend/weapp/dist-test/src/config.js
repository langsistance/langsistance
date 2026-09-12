"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.STORAGE_KEYS = exports.API_BASE = void 0;
/**
 * 运行时配置单一出口。构建期由 config/index.ts 的 defineConstants 注入
 * TARO_APP_API_BASE（编译命令可传 TARO_APP_API_BASE=... 覆盖）。
 */
exports.API_BASE = process.env.TARO_APP_API_BASE || 'http://127.0.0.1:7777';
exports.STORAGE_KEYS = {
    wxToken: 'copiioai_wx_token',
    userId: 'copiioai_user_id',
};
//# sourceMappingURL=config.js.map