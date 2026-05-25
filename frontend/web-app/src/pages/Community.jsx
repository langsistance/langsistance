import { useState, useEffect, useCallback } from 'react'
import { queryPublicKnowledge, copyKnowledge } from '../services/api'
import Pagination from '../components/Pagination'
import { KNOWLEDGE_LIST_PAGE_SIZE } from '../utils/appUiConfig'

function KnowledgeDetailModal({ item, onClose, onCopy, copying }) {
  const date = item.update_time
    ? new Date(item.update_time).toLocaleString('zh-CN')
    : ''

  return (
    <div className="modal">
      <div className="modal-overlay" onClick={onClose} />
      <div className="modal-content">
        <div className="modal-header">
          <h2>知识库详情</h2>
          <button className="modal-close-btn" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <div className="modal-body">
          <div className="metadata-section">
            <h4>知识内容</h4>
            <div className="detail-item">
              <strong>问题：</strong>{item.question}
            </div>
            {item.answer && (
              <div className="detail-item">
                <strong>答案：</strong>{item.answer}
              </div>
            )}
            {item.description && (
              <div className="detail-item">
                <strong>描述：</strong>{item.description}
              </div>
            )}
          </div>
          <div className="metadata-section">
            <h4>基本信息</h4>
            {item.extra_info?.email && (
              <div className="detail-item">
                <strong>📧 来自：</strong>{item.extra_info.email}
              </div>
            )}
            {date && (
              <div className="detail-item">
                <strong>更新时间：</strong>{date}
              </div>
            )}
          </div>
        </div>
        <div className="modal-footer">
          <button
            className="btn btn-primary"
            onClick={() => onCopy(item.id)}
            disabled={copying}
          >{copying ? '复制中...' : '添加到我的知识库'}</button>
          <button className="btn btn-secondary" onClick={onClose}>关闭</button>
        </div>
      </div>
    </div>
  )
}

export default function Community() {
  const [items, setItems] = useState([])
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [selected, setSelected] = useState(null)
  const [copying, setCopying] = useState(false)
  const PAGE_SIZE = KNOWLEDGE_LIST_PAGE_SIZE

  const [debouncedSearch, setDebouncedSearch] = useState(search)
  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search), 300)
    return () => clearTimeout(t)
  }, [search])

  const load = useCallback(async () => {
    try {
      const res = await queryPublicKnowledge({ search: debouncedSearch, page, limit: PAGE_SIZE })
      setItems(res.data || [])
      setTotal(res.total || 0)
    } catch (e) { console.error(e) }
  }, [debouncedSearch, page])

  useEffect(() => { load() }, [load])

  async function handleCopy(id) {
    setCopying(true)
    try {
      await copyKnowledge({ knowledgeId: id })
      alert('已添加到我的知识库')
      setSelected(null)
    } catch (e) {
      alert('复制失败：' + e.message)
    } finally {
      setCopying(false)
    }
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="page active">
      <div className="page-header">
        <div>
          <h1>社区</h1>
          <p>探索其他用户分享的知识库</p>
        </div>
      </div>

      <div className="page-content">
        <div className="knowledge-search">
          <input
            type="text"
            className="knowledge-search-input"
            placeholder="搜索社区知识库..."
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1) }}
          />
        </div>

        {items.length === 0 ? (
          <div className="empty-state">
            <p>暂无公开知识库</p>
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
                  <span>📅 {item.update_time ? new Date(item.update_time).toLocaleDateString('zh-CN') : ''}</span>
                </div>
                <div className="share-card-actions">
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={(e) => { e.stopPropagation(); handleCopy(item.id) }}
                  >添加到我的知识库</button>
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
