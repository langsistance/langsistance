import Taro from '@tarojs/taro'
import { request } from './api'
import { STORAGE_KEYS } from '../config'

export interface WechatLoginResult {
  token: string
  user_id: number
}

/**
 * 微信一键登录（决策 D3）：
 * wx.login → code → POST /auth/wechat → wx_ token 落本地。
 * AppID 未配置/服务端不可达时抛错，由调用页给出可读提示。
 */
export async function wechatLogin(): Promise<WechatLoginResult> {
  const login = await Taro.login()
  if (!login.code) {
    throw new Error('获取微信登录凭证失败，请重试')
  }
  const body = await request<WechatLoginResult>('/auth/wechat', {
    method: 'POST',
    data: { code: login.code },
    auth: false,
  })
  Taro.setStorageSync(STORAGE_KEYS.wxToken, body.token)
  Taro.setStorageSync(STORAGE_KEYS.userId, String(body.user_id))
  return body
}

export function isLoggedIn(): boolean {
  return Boolean(Taro.getStorageSync(STORAGE_KEYS.wxToken))
}

export function logout(): void {
  Taro.removeStorageSync(STORAGE_KEYS.wxToken)
  Taro.removeStorageSync(STORAGE_KEYS.userId)
}
