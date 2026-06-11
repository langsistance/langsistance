'use client'

import { useState, useEffect } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { getSceneKnowledge } from '@/services/api'

interface SceneInfo {
  id: number
  name: string
  description: string
  subscribed: boolean
  knowledge_count: number
}

interface SceneKnowledgeItem {
  id: number
  question: string
  description: string
}

interface SceneCardProps {
  scene: SceneInfo
  onToggle: (sceneId: number, subscribed: boolean) => void
}

export default function SceneCard({ scene, onToggle }: SceneCardProps) {
  const { t, lang } = useI18n()
  const [expanded, setExpanded] = useState(false)
  const [knowledgeItems, setKnowledgeItems] = useState<SceneKnowledgeItem[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (expanded && knowledgeItems.length === 0) {
      setLoading(true)
      getSceneKnowledge(scene.id)
        .then((res) => {
          setKnowledgeItems(res.knowledge || [])
        })
        .catch(() => {})
        .finally(() => setLoading(false))
    }
  }, [expanded, scene.id, knowledgeItems.length])

  const displayItems = expanded ? knowledgeItems : knowledgeItems.slice(0, 3)

  return (
    <div className="scene-card">
      <div className="scene-card-header">
        <div className="scene-card-info">
          <span className="scene-card-icon">📦</span>
          <div>
            <div className="scene-card-name">{scene.name}</div>
            <div className="scene-card-desc">{scene.description}</div>
          </div>
        </div>
        <div className="scene-card-actions">
          <span className={`scene-card-badge ${scene.subscribed ? 'enabled' : 'disabled'}`}>
            {scene.subscribed ? t('knowledge.sceneEnabled') : t('knowledge.sceneDisabled')}
          </span>
          <button
            className={`btn btn-sm ${scene.subscribed ? 'btn-outline' : 'btn-primary'}`}
            onClick={() => onToggle(scene.id, !scene.subscribed)}
          >
            {scene.subscribed
              ? (lang === 'en' ? 'Disable' : '取消启用')
              : (lang === 'en' ? 'Enable' : '启用')}
          </button>
        </div>
      </div>

      {knowledgeItems.length > 0 && (
        <div className="scene-card-body">
          <p className="scene-card-examples-label">
            {t('knowledge.sceneExampleQuestions')}
          </p>
          <ul className="scene-card-examples">
            {displayItems.map((item) => (
              <li key={item.id} className="scene-card-example-item">
                {item.question || item.description}
              </li>
            ))}
          </ul>

          {knowledgeItems.length > 3 && !expanded && (
            <button
              className="btn btn-link scene-card-expand-btn"
              onClick={() => setExpanded(true)}
            >
              {lang === 'en'
                ? `View all (${knowledgeItems.length})`
                : `查看全部 (${knowledgeItems.length})`}
            </button>
          )}
          {expanded && (
            <button
              className="btn btn-link scene-card-expand-btn"
              onClick={() => setExpanded(false)}
            >
              {t('knowledge.sceneCollapse')}
            </button>
          )}
        </div>
      )}

      {expanded && loading && (
        <div className="scene-card-body">
          <p className="scene-card-loading">{t('common.loading')}</p>
        </div>
      )}
    </div>
  )
}
