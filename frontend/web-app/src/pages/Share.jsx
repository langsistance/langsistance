import { useState, useEffect, useCallback } from 'react'
import {
  queryKnowledgeShares,
  getUserSharedKnowledge,
  authorizeKnowledgeAccess,
  cancelKnowledgeShare,
  handleKnowledgeShare,
  queryKnowledge,
} from '../services/api'
import Pagination from '../components/Pagination'

const PAGE_SIZE = 10

const STATUS_MAP = {
  1: { text: '待处理', cls: 'pending' },
  2: { text: '已拒绝', cls: 'rejected' },
  3: { text: '已接受', cls: 'accepted' },
  4: { text: '已撤销', cls: 'rejected' },
}

function CreateShareModal({ onClose, onSave }) {
  const [recipient, setRecipient] = useState('')
  const [knowledgeIds, setKnowledgeIds] = useState([])
  const [allKnowledge, setAllKnowledge] = useState([])
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    queryKnowledge({ page: 1, limit: 100 })
      .then((res) => setAllKnowledge(res.data || res.items || res.knowledge || []))
      .catch(() => {})
  }, [])

  function toggleId(id) {
    setKnowledgeIds((ids) =>
      ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]
    )
  }

  async function submit(e) {
    e.preventDefault()
    setError('')
    if (knowledgeIds.length === 0) { setError('请选择要分享的知识'); return }
    setSaving(true)
    try {
      await onSave({ email: recipient, knowledgeIds })
      onClose()
    } catch (err) {
      setError('分享失败：' + err.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal">
      <div className="modal-overlay" onClick={onClose} />
      <div className="modal-content">
        <div className="modal-header">
          <h2>创建分享</h2>
          <button className="modal-close-btn" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
          <div className="modal-body" style={{ flex: 1, overflowY: 'auto' }}>
            <div className="form-group">
              <label>分享给（用户邮箱）</label>
              <input
                type="email"
                className="form-input"
                placeholder="输入接收人邮箱..."
                value={recipient}
                onChange={(e) => setRecipient(e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label>选择要分享的知识</label>
              <div className="api-select-list">
                {allKnowledge.length === 0 && (
                  <p style={{ fontSize: 14, color: 'var(--color-text-secondary)', padding: 8 }}>暂无知识，请先创建知识</p>
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
            <button type="button" className="btn btn-secondary" onClick={onClose}>取消</button>
            <button type="submit" className="btn btn-primary" disabled={saving}>{saving ? '发送中...' : '发送分享'}</button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function Share() {
  const [tab, setTab] = useState('sent')
  const [sent, setSent] = useState([])
  const [sentPage, setSentPage] = useState(1)
  const [sentTotal, setSentTotal] = useState(0)
  const [received, setReceived] = useState([])
  const [receivedPage, setReceivedPage] = useState(1)
  const [receivedTotal, setReceivedTotal] = useState(0)
  const [showModal, setShowModal] = useState(false)

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

  function handleSentPageChange(p) {
    setSentPage(p)
    loadSent(p)
  }

  function handleReceivedPageChange(p) {
    setReceivedPage(p)
    loadReceived(p)
  }

  async function handleCreate({ email, knowledgeIds }) {
    for (const knowledgeId of knowledgeIds) {
      await authorizeKnowledgeAccess({ knowledgeId, email })
    }
    setSentPage(1)
    loadSent(1)
  }

  async function handleCancel(shareId) {
    if (!confirm('确认撤销分享？')) return
    try {
      await cancelKnowledgeShare({ share_id: shareId })
      loadSent(sentPage)
    } catch (e) { alert('撤销失败：' + e.message) }
  }

  async function handleAccept(shareId) {
    try {
      await handleKnowledgeShare({ share_id: shareId, action: 'accept' })
      loadReceived(receivedPage)
    } catch (e) { alert('接受失败：' + e.message) }
  }

  async function handleReject(shareId) {
    if (!confirm('确认拒绝该分享？')) return
    try {
      await handleKnowledgeShare({ share_id: shareId, action: 'reject' })
      loadReceived(receivedPage)
    } catch (e) { alert('拒绝失败：' + e.message) }
  }

  const sentTotalPages = Math.ceil(sentTotal / PAGE_SIZE)
  const receivedTotalPages = Math.ceil(receivedTotal / PAGE_SIZE)

  return (
    <div className="page active">
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>分享中心</h1>
            <p>分享知识库给其他用户，或接收他人分享的知识</p>
          </div>
          <button className="btn btn-primary" onClick={() => setShowModal(true)}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" />
              <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" /><line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
            </svg>
            创建分享
          </button>
        </div>
      </div>

      <div className="page-content">
        <div className="tabs-container">
          <button className={`tab-btn${tab === 'sent' ? ' active' : ''}`} onClick={() => setTab('sent')}>
            我的分享
          </button>
          <button className={`tab-btn${tab === 'received' ? ' active' : ''}`} onClick={() => setTab('received')}>
            收到的分享
          </button>
        </div>

        <div style={{ marginTop: 24 }}>
          {tab === 'sent' && (
            <>
              {sent.length === 0 ? (
                <div className="empty-state"><p>暂无分享记录</p></div>
              ) : (
                <div className="knowledge-list">
                  {sent.map((share, i) => {
                    const info = share.extra_info || {}
                    const status = STATUS_MAP[info.status] || { text: '未知', cls: 'pending' }
                    const date = info.share_update_time
                      ? new Date(info.share_update_time).toLocaleDateString('zh-CN')
                      : ''
                    return (
                      <div key={info.share_id || i} className="share-card">
                        <div className="share-card-header">
                          <div className="share-card-title">{share.question}</div>
                          <span className={`share-card-status ${status.cls}`}>{status.text}</span>
                        </div>
                        {info.to_user_email && (
                          <div className="share-card-info"><span>📧 {info.to_user_email}</span></div>
                        )}
                        {share.answer && (
                          <div className="share-card-message">"{share.answer}"</div>
                        )}
                        {date && (
                          <div className="share-card-meta"><span>📅 {date}</span></div>
                        )}
                        <div className="share-card-actions">
                          {info.status === 1 && (
                            <button
                              className="btn btn-secondary btn-sm"
                              style={{ color: '#D32F2F' }}
                              onClick={() => handleCancel(info.share_id)}
                            >撤销分享</button>
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
                <div className="empty-state"><p>暂无收到的分享</p></div>
              ) : (
                <div className="knowledge-list">
                  {received.map((share, i) => {
                    const info = share.extra_info || {}
                    const status = STATUS_MAP[info.status] || { text: '未知', cls: 'pending' }
                    const date = info.share_update_time
                      ? new Date(info.share_update_time).toLocaleDateString('zh-CN')
                      : ''
                    return (
                      <div key={info.share_id || i} className="share-card">
                        <div className="share-card-header">
                          <div className="share-card-title">{share.question}</div>
                          <span className={`share-card-status ${status.cls}`}>{status.text}</span>
                        </div>
                        {info.from_user_email && (
                          <div className="share-card-info"><span>📧 {info.from_user_email}</span></div>
                        )}
                        {share.answer && (
                          <div className="share-card-message">"{share.answer}"</div>
                        )}
                        {date && (
                          <div className="share-card-meta"><span>📅 {date}</span></div>
                        )}
                        <div className="share-card-actions">
                          {info.status === 1 && (
                            <>
                              <button className="btn btn-primary btn-sm" onClick={() => handleAccept(info.share_id)}>接受</button>
                              <button className="btn btn-secondary btn-sm" onClick={() => handleReject(info.share_id)}>拒绝</button>
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
    </div>
  )
}
