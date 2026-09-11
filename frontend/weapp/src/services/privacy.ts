import Taro from '@tarojs/taro'

/**
 * 隐私授权闸口（微信 2023-09 新规）。
 *
 * wx.chooseMessageFile 属隐私接口，未在公众平台《小程序用户隐私保护指引》
 * 声明「收集你选中的文件」时**接口直接禁用**（不是调用失败，是压根不存在）。
 * 故这里在真正调用前先确保用户已同意。
 *
 * 流程：调用方 await ensurePrivacy() → 若已同意立即返回；否则触发
 * onNeedPrivacyAuthorization，由页面把浮层显示出来，用户点同意后
 * resolve({ event: 'agree' }) 使本 Promise 兑现。
 */
type Resolver = (arg: { buttonId?: string; event: 'agree' | 'disagree' }) => void

let pendingResolve: Resolver | null = null
let listener: ((visible: boolean) => void) | null = null
let registered = false

/** App 级注册一次（app.ts 的 useLaunch 调用）。 */
export function registerPrivacyHandler(): void {
  if (registered) return
  registered = true
  if (typeof Taro.onNeedPrivacyAuthorization !== 'function') {
    // 基础库过低（<2.32.3）或非微信端：不做拦截，交给微信默认行为
    return
  }
  Taro.onNeedPrivacyAuthorization((resolve: any) => {
    pendingResolve = resolve
    listener?.(true)
  })
}

/** 页面订阅浮层显示状态；返回取消订阅函数。 */
export function subscribePrivacy(cb: (visible: boolean) => void): () => void {
  listener = cb
  return () => {
    if (listener === cb) listener = null
  }
}

/** 用户在浮层上的选择。agree 取自 <button open-type="agreePrivacyAuthorization">。 */
export function resolvePrivacy(agree: boolean, buttonId = ''): void {
  const resolve = pendingResolve
  pendingResolve = null
  listener?.(false)
  resolve?.({ buttonId, event: agree ? 'agree' : 'disagree' })
}

/**
 * 放弃本次等待（页面卸载/路由离开时调用）。
 * 不这样做的话 pendingResolve 会一直悬着，调用方的 await 永不兑现。
 * 用户没有做出选择，所以按"未同意"结算。
 */
export function abandonPrivacy(): void {
  const resolve = pendingResolve
  pendingResolve = null
  listener?.(false)
  resolve?.({ event: 'disagree' })
}

/**
 * 在调用任何隐私接口之前 await 这个。
 * 返回 true = 可以继续（已同意过，或本次刚同意）；
 * 返回 false = 用户拒绝，调用方必须中止，不要再去调隐私接口。
 *
 * 基础库不支持时返回 true —— 那种环境下微信不做拦截，硬拦反而会
 * 让功能在旧版本上不可用。
 */
export function ensurePrivacy(): Promise<boolean> {
  return new Promise((resolve) => {
    if (typeof Taro.requirePrivacyAuthorize !== 'function') {
      resolve(true)
      return
    }
    Taro.requirePrivacyAuthorize({
      success: () => resolve(true),
      fail: () => resolve(false),
    })
  })
}
