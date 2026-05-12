'use client'

import { useState, useEffect, useCallback } from 'react'
import { queryPublicKnowledge, copyKnowledge } from '@/services/api'
import { useI18n } from '@/lib/app-i18n'
import Pagination from '@/components/app/Pagination'

interface CommunityItem {
  id: number
  question: string
  answer: string
  update_time?: string
  extra_info?: { email?: string }
}

export default function Community() {
  const { t } = useI18n()
  const [items, setItems] = useState<CommunityItem[]>([])
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const PAGE_SIZE = 10

  const [debouncedSearch, setDebouncedSearch] = useState(search)
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(search), 300)
    return () => clearTimeout(timer)
  }, [search])

  const load = useCallback(async () => {
    try {
      const res = await queryPublicKnowledge({ search: debouncedSearch, page, limit: PAGE_SIZE })
      setItems(res.data || [])
      setTotal(res.total || 0)
    } catch (e) { console.error(e) }
  }, [debouncedSearch, page])

  useEffect(() => { load() }, [load])

  async function handleCopy(id: number) {
    try {
      await copyKnowledge({ knowledge_id: id })
      alert(t('community.copySuccess'))
    } catch (e) {
      alert(t('community.copyFailed') + ': ' + (e as Error).message)
    }
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="page active">
      <div className="page-header">
        <div>
          <h1>{t('community.title')}</h1>
          <p>{t('community.description')}</p>
        </div>
      </div>

      <div className="page-content">
        <div className="knowledge-search">
          <input
            type="text"
            className="knowledge-search-input"
            placeholder={t('community.searchTip')}
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1) }}
          />
        </div>

        {items.length === 0 ? (
          <div className="empty-state">
            <p>{t('community.noData')}</p>
          </div>
        ) : (
          <div className="knowledge-list">
            {items.map((item) => (
              <div key={item.id} className="share-card">
                <div className="share-card-header">
                  <div className="share-card-title">{item.question}</div>
                </div>
                {item.extra_info?.email && (
                  <div className="share-card-info">
                    <span>📧 {item.extra_info.email}</span>
                  </div>
                )}
                {item.answer && (
                  <div className="share-card-message">
                    &ldquo;{item.answer}&rdquo;
                  </div>
                )}
                <div className="share-card-meta">
                  <span>📅 {item.update_time ? new Date(item.update_time).toLocaleDateString('zh-CN') : ''}</span>
                </div>
                <div className="share-card-actions">
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={() => handleCopy(item.id)}
                  >{t('share.downloadKnowledge')}</button>
                </div>
              </div>
            ))}
          </div>
        )}

        {totalPages > 1 && (
          <Pagination page={page} totalPages={totalPages} onChange={setPage} />
        )}
      </div>
    </div>
  )
}
