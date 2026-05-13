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

interface ExtraInfo {
  to_user_email?: string
  from_user_email?: string
  status?: number
  share_id?: number
  share_update_time?: string
}

interface ShareItem {
  question: string
  answer?: string
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
  const [received, setReceived] = useState<ShareItem[]>([])
  const [showModal, setShowModal] = useState(false)

  const loadSent = useCallback(async () => {
    try {
      const res = await getUserSharedKnowledge({})
      setSent(res.data || [])
    } catch (e) { console.error(e) }
  }, [])

  const loadReceived = useCallback(async () => {
    try {
      const res = await queryKnowledgeShares({})
      setReceived(res.data || [])
    } catch (e) { console.error(e) }
  }, [])

  useEffect(() => { loadSent(); loadReceived() }, [loadSent, loadReceived])

  async function handleCreate({ email, knowledgeIds }: { email: string; knowledgeIds: number[] }) {
    for (const knowledgeId of knowledgeIds) {
      await authorizeKnowledgeAccess({ knowledgeId, email })
    }
    loadSent()
  }

  async function handleCancel(shareId: number) {
    if (!confirm(t('confirmations.cancelShare'))) return
    try {
      await cancelKnowledgeShare({ share_id: shareId })
      loadSent()
    } catch (e) { alert((e as Error).message) }
  }

  async function handleAccept(shareId: number) {
    try {
      await handleKnowledgeShare({ share_id: shareId, action: 'accept' })
      loadReceived()
    } catch (e) { alert((e as Error).message) }
  }

  async function handleReject(shareId: number) {
    if (!confirm(t('confirmations.refuseShare'))) return
    try {
      await handleKnowledgeShare({ share_id: shareId, action: 'reject' })
      loadReceived()
    } catch (e) { alert((e as Error).message) }
  }

  const list = tab === 'sent' ? sent : received

  const statusText = (status: number | undefined) => {
    if (status === 1) return t('common.pending')
    if (status === 2) return lang === 'en' ? 'Refused' : '已拒绝'
    if (status === 3) return lang === 'en' ? 'Accepted' : '已接受'
    if (status === 4) return lang === 'en' ? 'Cancelled' : '已撤销'
    return ''
  }

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
          {list.length === 0 ? (
            <div className="empty-state">
              <p>{tab === 'sent' ? t('share.noSharedDesc') : t('share.noReceived')}</p>
            </div>
          ) : (
            <div className="share-list">
              {list.map((share, i) => {
                const info = share.extra_info || {}
                const statusCls = STATUS_CLASS[info.status ?? 0] || 'pending'
                const email = tab === 'sent' ? info.to_user_email : info.from_user_email
                const date = info.share_update_time
                  ? new Date(info.share_update_time).toLocaleDateString(lang === 'en' ? 'en-US' : 'zh-CN')
                  : ''
                return (
                  <div key={info.share_id ?? i} className="share-card">
                    <div className="share-card-header">
                      <div className="share-card-title">{share.question}</div>
                      <span className={`share-card-status ${statusCls}`}>{statusText(info.status)}</span>
                    </div>
                    {email && (
                      <div className="share-card-info">
                        <span>📧 {email}</span>
                      </div>
                    )}
                    {share.answer && (
                      <div className="share-card-message">&ldquo;{share.answer}&rdquo;</div>
                    )}
                    {date && (
                      <div className="share-card-meta">
                        <span>📅 {date}</span>
                      </div>
                    )}
                    <div className="share-card-actions">
                      {tab === 'sent' && info.status === 1 && (
                        <button
                          className="btn btn-secondary btn-sm"
                          style={{ color: '#D32F2F' }}
                          onClick={() => handleCancel(info.share_id!)}
                        >{t('share.unshare')}</button>
                      )}
                      {tab === 'received' && info.status === 1 && (
                        <>
                          <button className="btn btn-primary btn-sm" onClick={() => handleAccept(info.share_id!)}>{t('common.accept')}</button>
                          <button className="btn btn-secondary btn-sm" onClick={() => handleReject(info.share_id!)}>{t('common.refuse')}</button>
                        </>
                      )}
                    </div>
                  </div>
                )
              })}
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
