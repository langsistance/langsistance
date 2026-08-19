'use client'

import { useEffect } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { getUserSceneStatus, updateUserScenes, markOnboarded } from '@/services/api'

// 新用户静默 onboarding（替代原场景选择弹窗）：
// 首次进入时自动订阅第一个场景并写入数据库，随后标记 onboarded。
// 全程无 UI；onboarded 已置位或接口失败时不重复写入，下次进入再试。
export default function SceneAutoOnboard() {
  const { lang } = useI18n()

  useEffect(() => {
    if (typeof window === 'undefined') return

    getUserSceneStatus(lang)
      .then(async (res: any) => {
        if (res.onboarded) return
        const first = (res.scenes || [])[0]
        if (!first) return
        try {
          await updateUserScenes([first.id])
          await markOnboarded()
        } catch {
          // 订阅失败则保持未 onboarded，下次进入重试；不阻塞页面
        }
      })
      .catch(() => {})
  }, [lang])

  return null
}
