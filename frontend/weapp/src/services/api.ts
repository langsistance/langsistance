import Taro from '@tarojs/taro'
import { API_BASE, STORAGE_KEYS } from '../config'

/** 统一请求封装：自动带 wx_ Bearer token；401 清凭证并跳登录页。 */
export interface ApiResult {
  success?: boolean
  [key: string]: any
}

/**
 * 凭证失效的统一处置：清本地 + 回登录页 + 抛出可见错误。
 * 抽出来是因为 Taro.uploadFile / Taro.downloadFile 走不到 request()，
 * 但同样需要这套处置。
 */
export function clearAuthAndRedirect(message = '登录已失效'): never {
  Taro.removeStorageSync(STORAGE_KEYS.wxToken)
  Taro.removeStorageSync(STORAGE_KEYS.userId)
  Taro.navigateTo({ url: '/pages/login/index' })
  throw new Error(message)
}

export async function request<T = ApiResult>(
  path: string,
  options: {
    method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
    data?: Record<string, any>
    auth?: boolean // 默认 true
  } = {},
): Promise<T> {
  const { method = 'GET', data, auth = true } = options
  const header: Record<string, string> = {
    'content-type': 'application/json',
  }
  if (auth) {
    const token = Taro.getStorageSync(STORAGE_KEYS.wxToken)
    if (token) {
      header.Authorization = `Bearer ${token}`
    }
  }
  const resp = await Taro.request({
    url: `${API_BASE}${path}`,
    method,
    data,
    header,
    timeout: 30000,
  })
  const body = resp.data as any
  if (resp.statusCode === 401) {
    clearAuthAndRedirect((body && (body.detail || body.message)) || '登录已失效')
  }
  if (resp.statusCode >= 400) {
    const detail = body && (body.detail || body.message)
    throw new Error(
      typeof detail === 'string' ? detail : `请求失败(${resp.statusCode})`,
    )
  }
  return body as T
}

/** 后端消息体可能把 detail 包成对象 —— 通用取错误文案 */
export function errorText(err: any, fallback = '操作失败，请稍后重试'): string {
  const detail = err && (err.detail || err.message)
  if (typeof detail === 'string') return detail
  if (detail && typeof detail.error === 'string') return detail.error
  if (err && typeof err.message === 'string') return err.message
  return fallback
}
