'use client'

import { useEffect, useMemo, useState } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { pickLang } from '@/lib/bilingual'
import {
  getUserSceneStatus,
  getSceneKnowledge,
  getPublicAvailableScenes,
  getPublicSceneKnowledge,
} from '@/services/api'

/**
 * Persistent scene hint block — lists subscribed scenes and their smart-QA /
 * deep-research capabilities.  Shared by the chat page and the results page
 * sidebar so both keep the identical pre-conversation context panel.
 */
export default function SceneHint() {
  const { t, lang } = useI18n()
  const [enabledScenes, setEnabledScenes] = useState<any[]>([])
  const [sceneSmartQA, setSceneSmartQA] = useState<{name: string, desc: string}[]>([])
  const [sceneDeepResearch, setSceneDeepResearch] = useState<{name: string, desc: string}[]>([])

  useEffect(() => {
    getUserSceneStatus(lang)
      .then(async (res) => {
        const subscribed = (res.scenes || []).filter((s: any) => s.subscribed)
        setEnabledScenes(subscribed)
        const smartQA: {name: string, desc: string}[] = []
        const deepResearch: {name: string, desc: string}[] = []
        for (const scene of subscribed) {
          try {
            const kr = await getSceneKnowledge(scene.id, lang)
            const items = kr.knowledge || []
            items.forEach((item: any) => {
              const example = {
                name: scene.name,
                desc: pickLang(item.description || item.question, lang),
              }
              if (item.type === 3) {
                deepResearch.push(example)
              } else {
                smartQA.push(example)
              }
            })
          } catch {}
        }
        setSceneSmartQA(smartQA)
        setSceneDeepResearch(deepResearch)
      })
      .catch(async () => {
        // Anonymous user — fall back to public scene endpoints (no auth)
        try {
          const pubRes = await getPublicAvailableScenes(lang)
          const allScenes = pubRes.scenes || []
          setEnabledScenes(allScenes)
          const smartQA: {name: string, desc: string}[] = []
          const deepResearch: {name: string, desc: string}[] = []
          for (const scene of allScenes) {
            try {
              const kr = await getPublicSceneKnowledge(scene.id, lang)
              const items = kr.knowledge || []
              items.forEach((item: any) => {
                const example = {
                  name: scene.name,
                  desc: pickLang(item.description || item.question, lang),
                }
                if (item.type === 3) {
                  deepResearch.push(example)
                } else {
                  smartQA.push(example)
                }
              })
            } catch {}
          }
          setSceneSmartQA(smartQA)
          setSceneDeepResearch(deepResearch)
        } catch {}
      })
  }, [lang])

  const groupedSmartQA = useMemo(() => {
    const map = new Map<string, {name: string, desc: string}[]>()
    for (const item of sceneSmartQA) {
      const existing = map.get(item.name)
      if (existing) {
        existing.push(item)
      } else {
        map.set(item.name, [item])
      }
    }
    return Array.from(map.entries())
  }, [sceneSmartQA])

  const groupedDeepResearch = useMemo(() => {
    const map = new Map<string, {name: string, desc: string}[]>()
    for (const item of sceneDeepResearch) {
      const existing = map.get(item.name)
      if (existing) {
        existing.push(item)
      } else {
        map.set(item.name, [item])
      }
    }
    return Array.from(map.entries())
  }, [sceneDeepResearch])

  if (enabledScenes.length === 0) return null

  return (
    <div className="scene-hint scene-hint-persistent">
      <div className="scene-hint-header">
        <span className="scene-hint-title">{t('chat.sceneHint')}</span>
      </div>
      <div className="scene-hint-scenes">
        {enabledScenes.map((scene, i) => (
          <span key={i} className="scene-hint-scene-tag">{scene.name}</span>
        ))}
      </div>

      {sceneSmartQA.length > 0 && (
        <div className="scene-hint-group scene-hint-group-smart">
          <div className="scene-hint-group-header">
            <span className="scene-hint-group-label">{t('chat.sceneSmartQA')}</span>
          </div>
          {groupedSmartQA.map(([sceneName, items]) => (
            <div key={sceneName} className="knowledge-group">
              <ul className="knowledge-group-list">
                {items.map((ex, i) => (
                  <li key={i} className="knowledge-group-item">
                    <span className="knowledge-group-item-dot" aria-hidden="true" />
                    <span className="knowledge-group-item-text">{ex.desc}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}

      {sceneDeepResearch.length > 0 && (
        <div className="scene-hint-group scene-hint-group-deep">
          <div className="scene-hint-group-header">
            <span className="scene-hint-group-label">{t('chat.sceneDeepResearch')}</span>
          </div>
          {groupedDeepResearch.map(([sceneName, items]) => (
            <div key={sceneName} className="knowledge-group">
              <ul className="knowledge-group-list">
                {items.map((ex, i) => (
                  <li key={i} className="knowledge-group-item">
                    <span className="knowledge-group-item-dot" aria-hidden="true" />
                    <span className="knowledge-group-item-text">{ex.desc}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
