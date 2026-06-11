'use client'

import { useState, useEffect, useCallback } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { getUserSceneStatus, updateUserScenes } from '@/services/api'
import SceneCard from '@/components/app/SceneCard'

const STORAGE_KEY = 'has_seen_onboarding'

export default function SceneOnboardingModal() {
  const { t, lang } = useI18n()
  const [visible, setVisible] = useState(false)
  const [scenes, setScenes] = useState<any[]>([])
  const [subscribedIds, setSubscribedIds] = useState<Set<number>>(new Set())
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // 检查是否首次访问
    if (typeof window === 'undefined') return
    if (localStorage.getItem(STORAGE_KEY)) return

    getUserSceneStatus()
      .then((res) => {
        const list = res.scenes || []
        setScenes(list)
        // 默认勾选所有场景（目前只有专利检索一个）
        const ids = new Set<number>()
        list.forEach((s: any) => {
          // 默认勾选：如果用户已订阅则保持，未订阅的新用户也默认勾选
          if (s.subscribed || list.length === 1) ids.add(s.id)
        })
        setSubscribedIds(ids)
        setVisible(true)
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const handleToggle = useCallback((sceneId: number, checked: boolean) => {
    setSubscribedIds((prev) => {
      const next = new Set(prev)
      if (checked) {
        next.add(sceneId)
      } else {
        next.delete(sceneId)
      }
      return next
    })
  }, [])

  async function handleConfirm() {
    const ids = Array.from(subscribedIds)
    try {
      await updateUserScenes(ids)
    } catch {}
    localStorage.setItem(STORAGE_KEY, '1')
    setVisible(false)
  }

  async function handleSkip() {
    try {
      await updateUserScenes([])
    } catch {}
    localStorage.setItem(STORAGE_KEY, '1')
    setVisible(false)
  }

  if (!visible || loading) return null

  return (
    <div className="modal">
      <div className="modal-overlay" />
      <div className="modal-content scene-onboarding-modal">
        <div className="modal-header">
          <h2>{lang === 'en' ? '🎉 Welcome to CopiioAI' : '🎉 欢迎来到 CopiioAI'}</h2>
        </div>
        <div className="modal-body">
          <p style={{ marginBottom: 'var(--spacing-lg)', color: 'var(--text-secondary)' }}>
            {lang === 'en'
              ? 'Choose your scenario to get started quickly:'
              : '选择你的使用场景，快速上手：'}
          </p>
          {scenes.map((scene) => (
            <SceneCard
              key={scene.id}
              scene={{ ...scene, subscribed: subscribedIds.has(scene.id) }}
              onToggle={handleToggle}
            />
          ))}
        </div>
        <div className="modal-footer">
          <button className="btn btn-outline" onClick={handleSkip}>
            {lang === 'en' ? "Skip for now" : '先不选，自己探索'}
          </button>
          <button className="btn btn-primary" onClick={handleConfirm}>
            {lang === 'en' ? 'Get Started' : '开始使用'}
          </button>
        </div>
      </div>
    </div>
  )
}
