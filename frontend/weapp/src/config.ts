/**
 * 运行时配置单一出口。构建期由 config/index.ts 的 defineConstants 注入
 * TARO_APP_API_BASE（编译命令可传 TARO_APP_API_BASE=... 覆盖）。
 */
export const API_BASE: string =
  process.env.TARO_APP_API_BASE || 'http://127.0.0.1:7777'

export const STORAGE_KEYS = {
  wxToken: 'copiioai_wx_token',
  userId: 'copiioai_user_id',
} as const
