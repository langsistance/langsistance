'use client'

import { useState, useEffect, useCallback } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { getUserSceneStatus, updateUserScenes, markOnboarded } from '@/services/api'
import SceneCard from '@/components/app/SceneCard'

export default function SceneOnboardingModal() {
  const { t, lang } = useI18n()
  const [visible, setVisible] = useState(false)
  const [scenes, setScenes] = useState<any[]>([])
  const [subscribedIds, setSubscribedIds] = useState<Set<number>>(new Set([1]))
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (typeof window === 'undefined') return

    getUserSceneStatus()
      .then((res) => {
        // 已 onboarded → 不显示弹窗，无闪烁
        if (res.onboarded) {
          setLoading(false)
          return
        }

        const list = res.scenes || []
        setScenes(list)
        if (list.length > 0) {
          const ids = new Set<number>()
          let hasAnySubscribed = false
          list.forEach((s: any) => {
            if (s.subscribed) {
              ids.add(s.id)
              hasAnySubscribed = true
            }
          })
          if (!hasAnySubscribed) {
            list.forEach((s: any) => ids.add(s.id))
          }
          setSubscribedIds(ids)
        }
        setVisible(true)
        setLoading(false)
      })
      .catch(() => {
        // API 失败时也显示弹窗，让用户至少能操作
        setVisible(true)
        setLoading(false)
      })
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
      await markOnboarded()
    } catch {}
    setVisible(false)
  }

  async function handleSkip() {
    try {
      await updateUserScenes([])
      await markOnboarded()
    } catch {}
    setVisible(false)
  }

  if (!visible) return null

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
          {loading && scenes.length === 0 && (
            <div className="scene-card">
              <div className="scene-card-header">
                <div className="scene-card-info">
                  <span className="scene-card-icon">📦</span>
                  <div>
                    <div className="scene-card-name">专利检索</div>
                    <div className="scene-card-desc">{t('common.loading')}</div>
                  </div>
                </div>
                <div className="scene-card-actions">
                  <div className="scene-toggle active">
                    <div className="scene-toggle-track" />
                    <div className="scene-toggle-thumb" />
                  </div>
                  <span className="scene-toggle-label">{t('knowledge.sceneEnabled')}</span>
                </div>
              </div>
            </div>
          )}
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
