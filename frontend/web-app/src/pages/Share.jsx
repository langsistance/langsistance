import { useState, useEffect, useCallback } from 'react'
import {
  queryKnowledgeShares,
  getUserSharedKnowledge,
  authorizeKnowledgeAccess,
  cancelKnowledgeShare,
  handleKnowledgeShare,
  queryKnowledge,
} from '../services/api'

function CreateShareModal({ onClose, onSave }) {
  const [recipient, setRecipient] = useState('')
  const [knowledgeIds, setKnowledgeIds] = useState([])
  const [allKnowledge, setAllKnowledge] = useState([])

  useEffect(() => {
    queryKnowledge({ page: 1, limit: 100 })
      .then((res) => setAllKnowledge(res.items || res.knowledge || []))
      .catch(() => {})
  }, [])

  function toggleId(id) {
    setKnowledgeIds((ids) =>
      ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]
    )
  }

  async function submit(e) {
    e.preventDefault()
    await onSave({ recipient_email: recipient, knowledge_ids: knowledgeIds })
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-slate-800 rounded-xl p-6 w-full max-w-lg border border-slate-700">
        <h3 className="text-base font-semibold mb-4">创建分享</h3>
        <form onSubmit={submit} className="flex flex-col gap-3">
          <input
            placeholder="接收人邮箱"
            type="email"
            value={recipient}
            onChange={(e) => setRecipient(e.target.value)}
            required
            className="bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-teal-500"
          />
          <div className="max-h-48 overflow-y-auto flex flex-col gap-1">
            {allKnowledge.map((k) => (
              <label key={k.id} className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={knowledgeIds.includes(k.id)}
                  onChange={() => toggleId(k.id)}
                  className="accent-teal-600"
                />
                {k.question}
              </label>
            ))}
          </div>
          <div className="flex justify-end gap-2 mt-2">
            <button type="button" onClick={onClose} className="px-4 py-2 text-sm text-slate-300 hover:text-white">
              取消
            </button>
            <button type="submit" className="px-4 py-2 bg-teal-600 hover:bg-teal-500 text-sm text-white rounded-lg">
              发送分享
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function Share() {
  const [tab, setTab] = useState('sent') // 'sent' | 'received'
  const [sent, setSent] = useState([])
  const [received, setReceived] = useState([])
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

  async function handleCreate(form) {
    try {
      await authorizeKnowledgeAccess(form)
      loadSent()
    } catch (e) {
      alert('分享失败：' + e.message)
    }
  }

  async function handleCancel(shareId) {
    if (!confirm('确认撤销分享？')) return
    await cancelKnowledgeShare({ share_id: shareId })
    loadSent()
  }

  async function handleAccept(shareId) {
    await handleKnowledgeShare({ share_id: shareId, action: 'accept' })
    loadReceived()
  }

  const list = tab === 'sent' ? sent : received

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">分享中心</h2>
          <p className="text-xs text-slate-500">分享知识库给其他用户</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="px-3 py-1.5 bg-teal-600 hover:bg-teal-500 text-xs text-white rounded-lg"
        >
          + 创建分享
        </button>
      </div>

      <div className="flex gap-2 px-4 py-3">
        {['sent', 'received'].map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-3 py-1.5 text-xs rounded-lg transition-colors ${
              tab === t
                ? 'bg-teal-600 text-white'
                : 'bg-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            {t === 'sent' ? '我的分享' : '收到的分享'}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto px-4 flex flex-col gap-2">
        {list.length === 0 && (
          <p className="text-slate-500 text-sm text-center mt-8">暂无记录</p>
        )}
        {list.map((share) => (
          <div key={share.id} className="bg-slate-800 rounded-lg p-4">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-slate-200">
                  {tab === 'sent' ? `→ ${share.recipient_email}` : `← ${share.sender_email}`}
                </p>
                <p className="text-xs text-slate-400 mt-0.5">
                  {share.knowledge_count || 0} 条知识 · {share.create_time?.slice(0, 10)}
                </p>
              </div>
              {tab === 'sent' ? (
                <button
                  onClick={() => handleCancel(share.id)}
                  className="text-xs text-slate-400 hover:text-red-400"
                >
                  撤销
                </button>
              ) : share.status !== 'accepted' ? (
                <button
                  onClick={() => handleAccept(share.id)}
                  className="text-xs text-teal-400 hover:text-teal-300"
                >
                  接受
                </button>
              ) : (
                <span className="text-xs text-slate-500">已接受</span>
              )}
            </div>
          </div>
        ))}
      </div>

      {showModal && (
        <CreateShareModal onClose={() => setShowModal(false)} onSave={handleCreate} />
      )}
    </div>
  )
}
