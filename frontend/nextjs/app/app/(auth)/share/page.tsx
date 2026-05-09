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

interface ShareItem {
  id: number
  recipient_email?: string
  sender_email?: string
  status?: string
  create_time?: string
  knowledge_count?: number
}

interface KnowledgeItem {
  id: number
  question: string
}

function CreateShareModal({ onClose, onSave }: {
  onClose: () => void
  onSave: (form: { recipient_email: string; knowledge_ids: number[] }) => Promise<void>
}) {
  const { t, lang } = useI18n()
  const [recipient, setRecipient] = useState('')
  const [knowledgeIds, setKnowledgeIds] = useState<number[]>([])
  const [allKnowledge, setAllKnowledge] = useState<KnowledgeItem[]>([])

  useEffect(() => {
    queryKnowledge({ page: 1, limit: 100 })
      .then((res) => setAllKnowledge(res.items || res.knowledge || []))
      .catch(() => {})
  }, [])

  function toggleId(id: number) {
    setKnowledgeIds((ids) =>
      ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]
    )
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    await onSave({ recipient_email: recipient, knowledge_ids: knowledgeIds })
    onClose()
  }

  return (
    <div className="modal">
      <div className="modal-overlay" onClick={onClose} />
      <div className="modal-content">
        <div className="modal-header">
          <h2>{t('share.create')}</h2>
          <button className="modal-close-btn" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <form onSubmit={submit}>
          <div className="modal-body">
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
          </div>
          <div className="modal-footer">
            <button type="button" className="btn btn-secondary" onClick={onClose}>{t('common.cancel')}</button>
            <button type="submit" className="btn btn-primary">{t('share.sendShare')}</button>
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
  const [received, setReceived] = useState<ShareItem[]>([])
  const [showModal, setShowModal] = useState(false)

  const loadSent = useCallback(async () => {
    try {
      const res = await queryKnowledgeShares({})
      setSent(res.shares || [])
    } catch (e) { console.error(e) }
  }, [])

  const loadReceived = useCallback(async () => {
    try {
      const res = await getUserSharedKnowledge({})
      setReceived(res.shares || [])
    } catch (e) { console.error(e) }
  }, [])

  useEffect(() => { loadSent(); loadReceived() }, [loadSent, loadReceived])

  async function handleCreate(form: { recipient_email: string; knowledge_ids: number[] }) {
    try {
      await authorizeKnowledgeAccess(form)
      loadSent()
    } catch (e) {
      alert((lang === 'en' ? 'Share failed: ' : '分享失败：') + (e as Error).message)
    }
  }

  async function handleCancel(shareId: number) {
    if (!confirm(t('confirmations.cancelShare'))) return
    await cancelKnowledgeShare({ share_id: shareId })
    loadSent()
  }

  async function handleAccept(shareId: number) {
    await handleKnowledgeShare({ share_id: shareId, action: 'accept' })
    loadReceived()
  }

  const list = tab === 'sent' ? sent : received

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
              <circle cx="18" cy="5" r="3" />
              <circle cx="6" cy="12" r="3" />
              <circle cx="18" cy="19" r="3" />
              <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
              <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
            </svg>
            {t('share.create')}
          </button>
        </div>
      </div>

      <div className="page-content">
        <div className="tabs-container">
          <button
            className={`tab-btn${tab === 'sent' ? ' active' : ''}`}
            onClick={() => setTab('sent')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
              <polyline points="22 4 12 14.01 9 11.01" />
            </svg>
            {t('share.my')}
          </button>
          <button
            className={`tab-btn${tab === 'received' ? ' active' : ''}`}
            onClick={() => setTab('received')}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
              <path d="M3 15v4a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-4" />
            </svg>
            {t('share.received')}
          </button>
        </div>

        <div style={{ marginTop: 24 }}>
          {list.length === 0 ? (
            <div className="empty-state">
              <p>{tab === 'sent' ? t('share.noSharedDesc') : t('share.noReceived')}</p>
            </div>
          ) : (
            <div className="share-list">
              {list.map((share) => (
                <div key={share.id} className="share-card">
                  <div className="share-card-header">
                    <div>
                      <p className="share-card-title">
                        {tab === 'sent'
                          ? `${lang === 'en' ? 'To' : '发送给'}: ${share.recipient_email}`
                          : `${lang === 'en' ? 'From' : '来自'}: ${share.sender_email}`}
                      </p>
                    </div>
                    <span className={`share-card-status ${share.status || 'pending'}`}>
                      {share.status === 'accepted'
                        ? (lang === 'en' ? 'Accepted' : '已接受')
                        : share.status === 'rejected'
                        ? (lang === 'en' ? 'Rejected' : '已拒绝')
                        : t('common.pending')}
                    </span>
                  </div>
                  <div className="share-card-info">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
                      <line x1="16" y1="2" x2="16" y2="6" />
                      <line x1="8" y1="2" x2="8" y2="6" />
                      <line x1="3" y1="10" x2="21" y2="10" />
                    </svg>
                    {share.create_time?.slice(0, 10)} · {share.knowledge_count || 0} {lang === 'en' ? 'items' : '条知识'}
                  </div>
                  <div className="share-card-actions">
                    {tab === 'sent' ? (
                      <button
                        className="btn btn-secondary btn-sm"
                        style={{ color: '#D32F2F' }}
                        onClick={() => handleCancel(share.id)}
                      >{t('share.unshare')}</button>
                    ) : share.status !== 'accepted' ? (
                      <button
                        className="btn btn-primary btn-sm"
                        onClick={() => handleAccept(share.id)}
                      >{t('common.accept')}</button>
                    ) : (
                      <span style={{ fontSize: 14, color: 'var(--color-text-secondary)' }}>{lang === 'en' ? 'Accepted' : '已接受'}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {showModal && (
        <CreateShareModal onClose={() => setShowModal(false)} onSave={handleCreate} />
      )}
    </div>
  )
}
