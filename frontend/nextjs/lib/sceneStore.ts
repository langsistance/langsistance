/**
 * Scene mode persistence (seller line A2).
 *
 * Scene is UI state, NOT a route: switching 卖家安全台 ⇄ 专业工作台 never
 * changes the URL (design spec §3.2-1). The value is read at send time so
 * the streaming hook needs no prop drilling.
 */

export type SceneMode = 'seller' | 'pro'

const SCENE_KEY = 'copiioai_scene'

export function getSceneMode(): SceneMode {
  if (typeof window === 'undefined') return 'pro'
  try {
    return window.localStorage.getItem(SCENE_KEY) === 'seller' ? 'seller' : 'pro'
  } catch {
    return 'pro'
  }
}

export function setSceneMode(mode: SceneMode): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(SCENE_KEY, mode)
  } catch {
    /* storage unavailable — mode stays session-default */
  }
}
