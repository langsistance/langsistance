import { useState, useEffect, useCallback } from 'react'
import {
  queryKnowledge,
  createKnowledge,
  updateKnowledge,
  deleteKnowledge,
  queryTools,
} from '../services/api'
import Pagination from '../components/Pagination'

function KnowledgeModal({ item, tools, onClose, onSave }) {
  const [form, setForm] = useState(
    item ? { ...item } : { question: '', answer: '', description: '', tool_id: '', public: 0 }
  )
  const [saveError, setSaveError] = useState('')

  function set(key, val) {
    setForm((f) => ({ ...f, [key]: val }))
  }

  async function submit(e) {
    e.preventDefault()
    setSaveError('')
    try {
      await onSave(form)
      onClose()
    } catch (err) {
      setSaveError(err.message || '保存失败')
    }
  }

  return (
    <div className="modal">
      <div className="modal-overlay" onClick={onClose} />
      <div className="modal-content knowledge-editor-modal">
        <div className="modal-header">
          <h2>{item ? '编辑知识库' : '创建知识库'}</h2>
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
              <label>问题</label>
              <input
                className="form-input"
                placeholder="问题（AI 匹配时使用）"
                value={form.question}
                onChange={(e) => set('question', e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label>答案</label>
              <textarea
                className="form-textarea"
                placeholder="答案 / 工具调用说明"
                value={form.answer}
                onChange={(e) => set('answer', e.target.value)}
                required
                rows={4}
              />
            </div>
            <div className="form-group">
              <label>描述（可选）</label>
              <textarea
                className="form-textarea"
                placeholder="补充说明"
                value={form.description}
                onChange={(e) => set('description', e.target.value)}
                rows={2}
              />
            </div>
            <div className="form-group">
              <label>关联工具</label>
              <select
                className="form-input"
                value={String(form.tool_id)}
                onChange={(e) => set('tool_id', e.target.value)}
              >
                <option value="">无关联工具</option>
                {tools.map((tool) => (
                  <option key={tool.id} value={tool.id}>{tool.title}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={!!form.public}
                  onChange={(e) => set('public', e.target.checked ? 1 : 0)}
                />
                公开（在社区可见）
              </label>
            </div>
            {saveError && <p style={{ color: '#D32F2F', fontSize: 14, marginTop: -8 }}>{saveError}</p>}
          </div>
          <div className="modal-footer">
            <button type="button" className="btn btn-secondary" onClick={onClose}>取消</button>
            <button type="submit" className="btn btn-primary">保存</button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function Knowledge() {
  const [items, setItems] = useState([])
  const [tools, setTools] = useState([])
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [modal, setModal] = useState(null)
  const PAGE_SIZE = 10

  const [debouncedSearch, setDebouncedSearch] = useState(search)
  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search), 300)
    return () => clearTimeout(t)
  }, [search])

  const load = useCallback(async () => {
    try {
      const res = await queryKnowledge({ search: debouncedSearch, page, limit: PAGE_SIZE })
      const data = res.data
      setItems(Array.isArray(data) ? data : (data?.knowledge || []))
      setTotal(res.total || 0)
    } catch (e) { console.error(e) }
  }, [debouncedSearch, page])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    queryTools({})
      .then((res) => setTools((res.data || []).filter((t) => t.push === 2)))
      .catch(() => {})
  }, [])

  async function handleSave(form) {
    if (form.id) {
      await updateKnowledge(form)
    } else {
      await createKnowledge(form)
    }
    load()
  }

  async function handleDelete(id) {
    if (!confirm('确认删除？')) return
    await deleteKnowledge({ id })
    load()
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="page active">
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>知识库</h1>
            <p>管理 API 文档和使用说明</p>
          </div>
          <button className="btn btn-primary" onClick={() => setModal('create')}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            创建知识库
          </button>
        </div>
      </div>

      <div className="page-content">
        <div className="knowledge-search">
          <input
            type="text"
            className="knowledge-search-input"
            placeholder="搜索知识库..."
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1) }}
          />
        </div>

        {items.length === 0 ? (
          <div className="empty-state">
            <p>暂无知识库，点击右上角创建</p>
          </div>
        ) : (
          <div className="knowledge-list">
            {items.map((item) => (
              <div key={item.id} className="knowledge-card">
                <div className="knowledge-card-header">
                  <div style={{ flex: 1 }}>
                    <p className="knowledge-card-title">{item.question}</p>
                  </div>
                  <div className="knowledge-card-actions">
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => setModal(item)}
                    >编辑</button>
                    <button
                      className="btn btn-secondary btn-sm"
                      style={{ color: '#D32F2F' }}
                      onClick={() => handleDelete(item.id)}
                    >删除</button>
                  </div>
                </div>
                <p className="knowledge-card-content">{item.answer}</p>
                {Number(item.tool_id) > 0 && (
                  <div className="knowledge-card-apis">
                    <span className="knowledge-api-badge">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" />
                      </svg>
                      {tools.find((t) => t.id === Number(item.tool_id))?.title || `Tool #${item.tool_id}`}
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        <Pagination page={page} totalPages={totalPages} onChange={setPage} />
      </div>

      {modal && (
        <KnowledgeModal
          item={modal === 'create' ? null : modal}
          tools={tools}
          onClose={() => setModal(null)}
          onSave={handleSave}
        />
      )}
    </div>
  )
}
