import { useState, useEffect, useCallback } from 'react'
import { queryPublicKnowledge, copyKnowledge } from '../services/api'
import Pagination from '../components/Pagination'

export default function Community() {
  const [items, setItems] = useState([])
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const PAGE_SIZE = 10

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
    try {
      await copyKnowledge({ knowledge_id: id })
      alert('已复制到我的知识库')
    } catch (e) {
      alert('复制失败：' + e.message)
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
              <div key={item.id} className="community-card">
                <div className="community-card-header">
                  <p className="knowledge-card-title" style={{ flex: 1 }}>{item.question}</p>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => handleCopy(item.id)}
                  >复制</button>
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

        <Pagination page={page} totalPages={totalPages} onChange={setPage} />
      </div>
    </div>
  )
}
