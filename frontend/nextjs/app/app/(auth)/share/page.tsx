'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  queryKnowledgeShares,
  getUserSharedKnowledge,
  authorizeKnowledgeAccess,
  cancelKnowledgeShare,
  handleKnowledgeShare,
  queryKnowledge,
} from '@/services/api'
import { useI18n } from '@/lib/app-i18n'
import { getKnowledgeTypeBadge } from '@/lib/knowledgeTypeBadge'
import Pagination from '@/components/app/Pagination'
import KnowledgeDetailModal from '@/components/app/KnowledgeDetailModal'

const PAGE_SIZE = 10

interface ExtraInfo {
  to_user_email?: string
  from_user_email?: string
  status?: number
  share_id?: number
  share_update_time?: string
}

interface ShareItem {
  id?: number
  question: string
  answer?: string
  description?: string
  type?: number
  params?: string
  update_time?: string
  extra_info: ExtraInfo
}

interface KnowledgeItem {
  id: number
  question: string
}

const STATUS_CLASS: Record<number, string> = {
  1: 'pending',
  2: 'rejected',
  3: 'accepted',
  4: 'rejected',
}

function CreateShareModal({ onClose, onSave }: {
  onClose: () => void
  onSave: (form: { email: string; knowledgeIds: number[] }) => Promise<void>
}) {
  const { t } = useI18n()
  const [recipient, setRecipient] = useState('')
  const [knowledgeIds, setKnowledgeIds] = useState<number[]>([])
  const [allKnowledge, setAllKnowledge] = useState<KnowledgeItem[]>([])
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    queryKnowledge({ page: 1, limit: 100 })
      .then((res) => setAllKnowledge(res.data || res.items || res.knowledge || []))
      .catch(() => {})
  }, [])

  function toggleId(id: number) {
    setKnowledgeIds((ids) =>
      ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]
    )
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    if (knowledgeIds.length === 0) { setError(t('alerts.selectKnowledgeTips')); return }
    setSaving(true)
    try {
      await onSave({ email: recipient, knowledgeIds })
      onClose()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal">
      <div className="modal-overlay" onClick={onClose} />
      <div className="modal-content">
        <div className="modal-header">
          <h2>{t('share.create')}</h2>
          <button className="modal-close-btn" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
          <div className="modal-body" style={{ flex: 1, overflowY: 'auto' }}>
            <div className="form-group">
              <label>{t('share.shareWith')}</label>
              <input
                type="email"
                className="form-input"
                placeholder={t('share.emailDesc')}
                value={recipient}
                onChange={(e) => setRecipient(e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label>{t('share.chooseKnowledgeShare')}</label>
              <div className="api-select-list">
                {allKnowledge.length === 0 && (
                  <p style={{ fontSize: 14, color: 'var(--color-text-secondary)', padding: 8 }}>{t('share.noKnowledgeDesc')}</p>
                )}
                {allKnowledge.map((k) => (
                  <div key={k.id} className="api-select-item">
                    <input
                      type="checkbox"
                      id={`k-${k.id}`}
                      checked={knowledgeIds.includes(k.id)}
                      onChange={() => toggleId(k.id)}
                    />
                    <label htmlFor={`k-${k.id}`}>{k.question}</label>
                  </div>
                ))}
              </div>
            </div>
            {error && <p style={{ color: '#D32F2F', fontSize: 14 }}>{error}</p>}
          </div>
          <div className="modal-footer">
            <button type="button" className="btn btn-secondary" onClick={onClose}>{t('common.cancel')}</button>
            <button type="submit" className="btn btn-primary" disabled={saving}>{saving ? t('common.loading') : t('share.sendShare')}</button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function Share() {
  const { t, lang } = useI18n()
  const [tab, setTab] = useState('sent')
  const [sent, setSent] = useState<ShareItem[]>([])
  const [sentPage, setSentPage] = useState(1)
  const [sentTotal, setSentTotal] = useState(0)
  const [received, setReceived] = useState<ShareItem[]>([])
  const [receivedPage, setReceivedPage] = useState(1)
  const [receivedTotal, setReceivedTotal] = useState(0)
  const [showModal, setShowModal] = useState(false)
  const [selectedItem, setSelectedItem] = useState<ShareItem | null>(null)
  const [selectedType, setSelectedType] = useState<'sent' | 'received'>('sent')

  const loadSent = useCallback(async (page = 1) => {
    try {
      const res = await getUserSharedKnowledge({ limit: PAGE_SIZE, offset: (page - 1) * PAGE_SIZE })
      setSent(res.data || [])
      setSentTotal(res.total || 0)
    } catch (e) { console.error(e) }
  }, [])

  const loadReceived = useCallback(async (page = 1) => {
    try {
      const res = await queryKnowledgeShares({ limit: PAGE_SIZE, offset: (page - 1) * PAGE_SIZE })
      setReceived(res.data || [])
      setReceivedTotal(res.total || 0)
    } catch (e) { console.error(e) }
  }, [])

  useEffect(() => { loadSent(1); loadReceived(1) }, [loadSent, loadReceived])

  function handleSentPageChange(p: number) {
    setSentPage(p)
    loadSent(p)
  }

  function handleReceivedPageChange(p: number) {
    setReceivedPage(p)
    loadReceived(p)
  }

  async function handleCreate({ email, knowledgeIds }: { email: string; knowledgeIds: number[] }) {
    for (const knowledgeId of knowledgeIds) {
      await authorizeKnowledgeAccess({ knowledgeId, email })
    }
    setSentPage(1)
    loadSent(1)
  }

  async function handleCancel(shareId: number) {
    if (!confirm(t('confirmations.cancelShare'))) return
    try {
      await cancelKnowledgeShare({ share_id: shareId })
      loadSent(sentPage)
    } catch (e) { alert((e as Error).message) }
  }

  async function handleAccept(shareId: number) {
    try {
      await handleKnowledgeShare({ share_id: shareId, action: 'accept' })
      loadReceived(receivedPage)
    } catch (e) { alert((e as Error).message) }
  }

  async function handleReject(shareId: number) {
    if (!confirm(t('confirmations.refuseShare'))) return
    try {
      await handleKnowledgeShare({ share_id: shareId, action: 'reject' })
      loadReceived(receivedPage)
    } catch (e) { alert((e as Error).message) }
  }

  const statusText = (status: number | undefined) => {
    if (status === 1) return t('common.pending')
    if (status === 2) return lang === 'en' ? 'Refused' : '已拒绝'
    if (status === 3) return lang === 'en' ? 'Accepted' : '已接受'
    if (status === 4) return lang === 'en' ? 'Cancelled' : '已撤销'
    return ''
  }

  const sentTotalPages = Math.ceil(sentTotal / PAGE_SIZE)
  const receivedTotalPages = Math.ceil(receivedTotal / PAGE_SIZE)
  const selectedInfo = selectedItem?.extra_info || {}
  const selectedShareDate = selectedInfo.share_update_time
    ? new Date(selectedInfo.share_update_time).toLocaleString(lang === 'en' ? 'en-US' : 'zh-CN')
    : ''
  const selectedUpdateDate = selectedItem?.update_time
    ? new Date(selectedItem.update_time).toLocaleString(lang === 'en' ? 'en-US' : 'zh-CN')
    : ''

  return (
    <div className="page active">
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>{t('share.title')}</h1>
            <p>{t('share.description')}</p>
          </div>
          <button className="btn btn-primary" onClick={() => setShowModal(true)}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" />
              <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" /><line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
            </svg>
            {t('share.create')}
          </button>
        </div>
      </div>

      <div className="page-content">
        <div className="tabs-container">
          <button className={`tab-btn${tab === 'sent' ? ' active' : ''}`} onClick={() => setTab('sent')}>
            {t('share.my')}
          </button>
          <button className={`tab-btn${tab === 'received' ? ' active' : ''}`} onClick={() => setTab('received')}>
            {t('share.received')}
          </button>
        </div>

        <div style={{ marginTop: 24 }}>
          {tab === 'sent' && (
            <>
              {sent.length === 0 ? (
                <div className="empty-state"><p>{t('share.noSharedDesc')}</p></div>
              ) : (
                <div className="knowledge-list">
                  {sent.map((share, i) => {
                    const info = share.extra_info || {}
                    const statusCls = STATUS_CLASS[info.status ?? 0] || 'pending'
                    const typeBadge = getKnowledgeTypeBadge(share.type, lang, share.params)
                    const date = info.share_update_time
                      ? new Date(info.share_update_time).toLocaleDateString(lang === 'en' ? 'en-US' : 'zh-CN')
                      : ''
                    return (
                      <div key={info.share_id ?? i} className="share-card" style={{ cursor: 'pointer' }} onClick={() => { setSelectedItem(share); setSelectedType('sent') }}>
                        <div className="share-card-header">
                          <div className="share-card-title">{share.question}</div>
                          <div className="share-card-header-badges">
                            <span className={typeBadge.className}>{typeBadge.label}</span>
                            <span className={`share-card-status ${statusCls}`}>{statusText(info.status)}</span>
                          </div>
                        </div>
                        {info.to_user_email && (
                          <div className="share-card-info"><span>📧 {info.to_user_email}</span></div>
                        )}
                        {share.answer && (
                          <div className="share-card-message">&ldquo;{share.answer}&rdquo;</div>
                        )}
                        {date && (
                          <div className="share-card-meta"><span>📅 {date}</span></div>
                        )}
                        <div className="share-card-actions">
                          {info.status === 1 && (
                            <button
                              className="btn btn-secondary btn-sm"
                              style={{ color: '#D32F2F' }}
                              onClick={(e) => { e.stopPropagation(); handleCancel(info.share_id!) }}
                            >{t('share.unshare')}</button>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
              <Pagination page={sentPage} totalPages={sentTotalPages} onChange={handleSentPageChange} />
            </>
          )}

          {tab === 'received' && (
            <>
              {received.length === 0 ? (
                <div className="empty-state"><p>{t('share.noReceived')}</p></div>
              ) : (
                <div className="knowledge-list">
                  {received.map((share, i) => {
                    const info = share.extra_info || {}
                    const statusCls = STATUS_CLASS[info.status ?? 0] || 'pending'
                    const typeBadge = getKnowledgeTypeBadge(share.type, lang, share.params)
                    const date = info.share_update_time
                      ? new Date(info.share_update_time).toLocaleDateString(lang === 'en' ? 'en-US' : 'zh-CN')
                      : ''
                    return (
                      <div key={info.share_id ?? i} className="share-card" style={{ cursor: 'pointer' }} onClick={() => { setSelectedItem(share); setSelectedType('received') }}>
                        <div className="share-card-header">
                          <div className="share-card-title">{share.question}</div>
                          <div className="share-card-header-badges">
                            <span className={typeBadge.className}>{typeBadge.label}</span>
                            <span className={`share-card-status ${statusCls}`}>{statusText(info.status)}</span>
                          </div>
                        </div>
                        {info.from_user_email && (
                          <div className="share-card-info"><span>📧 {info.from_user_email}</span></div>
                        )}
                        {share.answer && (
                          <div className="share-card-message">&ldquo;{share.answer}&rdquo;</div>
                        )}
                        {date && (
                          <div className="share-card-meta"><span>📅 {date}</span></div>
                        )}
                        <div className="share-card-actions">
                          {info.status === 1 && (
                            <>
                              <button className="btn btn-primary btn-sm" onClick={(e) => { e.stopPropagation(); handleAccept(info.share_id!) }}>{t('common.accept')}</button>
                              <button className="btn btn-secondary btn-sm" onClick={(e) => { e.stopPropagation(); handleReject(info.share_id!) }}>{t('common.refuse')}</button>
                            </>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
              <Pagination page={receivedPage} totalPages={receivedTotalPages} onChange={handleReceivedPageChange} />
            </>
          )}
        </div>
      </div>

      {showModal && (
        <CreateShareModal onClose={() => setShowModal(false)} onSave={handleCreate} />
      )}

      {selectedItem && (
        <KnowledgeDetailModal
          item={selectedItem}
          onClose={() => setSelectedItem(null)}
          metadata={(
            <>
              {selectedType === 'sent' && selectedInfo.to_user_email && (
                <div className="detail-item">
                  <strong>📧 {lang === 'en' ? 'Shared with' : '分享给'}：</strong>{selectedInfo.to_user_email}
                </div>
              )}
              {selectedType === 'received' && selectedInfo.from_user_email && (
                <div className="detail-item">
                  <strong>📧 {lang === 'en' ? 'From' : '来自'}：</strong>{selectedInfo.from_user_email}
                </div>
              )}
              {selectedUpdateDate && (
                <div className="detail-item">
                  <strong>{t('share.knowledgeUpdateTime')}：</strong>{selectedUpdateDate}
                </div>
              )}
              {selectedShareDate && (
                <div className="detail-item">
                  <strong>{t('share.shareUpdateTime')}：</strong>{selectedShareDate}
                </div>
              )}
            </>
          )}
          footer={(
            <>
              {selectedType === 'sent' && selectedInfo.status === 1 && (
                <button
                  className="btn btn-secondary"
                  style={{ color: '#D32F2F' }}
                  onClick={() => { setSelectedItem(null); handleCancel(selectedInfo.share_id!) }}
                >{t('share.unshare')}</button>
              )}
              {selectedType === 'received' && selectedInfo.status === 1 && (
                <>
                  <button className="btn btn-primary" onClick={() => { setSelectedItem(null); handleAccept(selectedInfo.share_id!) }}>{t('common.accept')}</button>
                  <button className="btn btn-secondary" onClick={() => { setSelectedItem(null); handleReject(selectedInfo.share_id!) }}>{t('common.refuse')}</button>
                </>
              )}
            </>
          )}
        />
      )}
    </div>
  )
}
