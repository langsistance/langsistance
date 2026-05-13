'use client'

import { useState, useEffect, useCallback } from 'react'
import { queryPublicKnowledge, copyKnowledge } from '@/services/api'
import { useI18n } from '@/lib/app-i18n'
import Pagination from '@/components/app/Pagination'

interface CommunityItem {
  id: number
  question: string
  answer?: string
  description?: string
  update_time?: string
  extra_info?: { email?: string }
}

function KnowledgeDetailModal({ item, onClose, onCopy, copying }: {
  item: CommunityItem
  onClose: () => void
  onCopy: (id: number) => void
  copying: boolean
}) {
  const { t, lang } = useI18n()
  const date = item.update_time
    ? new Date(item.update_time).toLocaleString(lang === 'en' ? 'en-US' : 'zh-CN')
    : ''

  return (
    <div className="modal">
      <div className="modal-overlay" onClick={onClose} />
      <div className="modal-content">
        <div className="modal-header">
          <h2>{t('modals.knowledgeDetails.title')}</h2>
          <button className="modal-close-btn" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <div className="modal-body">
          <div className="metadata-section">
            <h4>{t('modals.knowledgeDetails.knowledgeContent')}</h4>
            <div className="detail-item">
              <strong>{t('knowledge.question')}：</strong>{item.question}
            </div>
            {item.answer && (
              <div className="detail-item">
                <strong>{t('knowledge.answer')}：</strong>{item.answer}
              </div>
            )}
            {item.description && (
              <div className="detail-item">
                <strong>{t('modals.knowledgeDetails.noDescription') ? (lang === 'en' ? 'Description' : '描述') : ''}：</strong>{item.description}
              </div>
            )}
          </div>
          <div className="metadata-section">
            <h4>{t('modals.knowledgeDetails.basicInfo')}</h4>
            {item.extra_info?.email && (
              <div className="detail-item">
                <strong>📧 {lang === 'en' ? 'From' : '来自'}：</strong>{item.extra_info.email}
              </div>
            )}
            {date && (
              <div className="detail-item">
                <strong>{t('share.knowledgeUpdateTime')}：</strong>{date}
              </div>
            )}
          </div>
        </div>
        <div className="modal-footer">
          <button
            className="btn btn-primary"
            onClick={() => onCopy(item.id)}
            disabled={copying}
          >{copying ? t('common.loading') : t('share.downloadKnowledge')}</button>
          <button className="btn btn-secondary" onClick={onClose}>{t('modals.close')}</button>
        </div>
      </div>
    </div>
  )
}

export default function Community() {
  const { t, lang } = useI18n()
  const [items, setItems] = useState<CommunityItem[]>([])
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [selected, setSelected] = useState<CommunityItem | null>(null)
  const [copying, setCopying] = useState(false)
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
    setCopying(true)
    try {
      await copyKnowledge({ knowledgeId: id })
      alert(t('community.copySuccess'))
      setSelected(null)
    } catch (e) {
      alert(t('community.copyFailed') + ': ' + (e as Error).message)
    } finally {
      setCopying(false)
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
              <div
                key={item.id}
                className="share-card"
                style={{ cursor: 'pointer' }}
                onClick={() => setSelected(item)}
              >
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
                  <span>📅 {item.update_time ? new Date(item.update_time).toLocaleDateString(lang === 'en' ? 'en-US' : 'zh-CN') : ''}</span>
                </div>
                <div className="share-card-actions">
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={(e) => { e.stopPropagation(); handleCopy(item.id) }}
                  >{t('share.downloadKnowledge')}</button>
                </div>
              </div>
            ))}
          </div>
        )}

        <Pagination page={page} totalPages={totalPages} onChange={setPage} />
      </div>

      {selected && (
        <KnowledgeDetailModal
          item={selected}
          onClose={() => setSelected(null)}
          onCopy={handleCopy}
          copying={copying}
        />
      )}
    </div>
  )
}
