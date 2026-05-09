'use client'

import { useState, useEffect, useCallback } from 'react'
import { queryPublicKnowledge, copyKnowledge } from '@/services/api'
import { useI18n } from '@/lib/app-i18n'

interface CommunityItem {
  id: number
  question: string
  answer: string
  user_email?: string
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
      setItems(res.items || res.knowledge || [])
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
              <div key={item.id} className="community-card">
                <div className="community-card-header">
                  <p className="knowledge-card-title" style={{ flex: 1 }}>{item.question}</p>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => handleCopy(item.id)}
                  >{t('common.copy')}</button>
                </div>
                <p className="knowledge-card-content">{item.answer}</p>
                {item.user_email && (
                  <p style={{ fontSize: 13, color: 'var(--color-text-secondary)', marginTop: 4 }}>
                    by {item.user_email}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}

        {totalPages > 1 && (
          <div className="pagination">
            <button
              className="pagination-btn"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
            >‹</button>
            <span className="pagination-info">{page} / {totalPages}</span>
            <button
              className="pagination-btn"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
            >›</button>
          </div>
        )}
      </div>
    </div>
  )
}
