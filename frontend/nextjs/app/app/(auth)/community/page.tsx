'use client'

import { useState, useEffect, useCallback } from 'react'
import { queryPublicKnowledge, copyKnowledge } from '@/services/api'

interface CommunityItem {
  id: number
  question: string
  answer: string
  user_email?: string
}

export default function Community() {
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
      alert('已复制到我的知识库')
    } catch (e) {
      alert('复制失败：' + (e as Error).message)
    }
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-slate-800">
        <h2 className="text-sm font-semibold text-slate-200">社区</h2>
        <p className="text-xs text-slate-500">探索其他用户分享的知识库</p>
      </div>

      <div className="px-4 py-3">
        <input
          placeholder="搜索社区知识库..."
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(1) }}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-teal-500"
        />
      </div>

      <div className="flex-1 overflow-y-auto px-4 flex flex-col gap-2">
        {items.map((item) => (
          <div key={item.id} className="bg-slate-800 rounded-lg p-4 flex items-start justify-between gap-3">
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-slate-200">{item.question}</p>
              <p className="text-xs text-slate-400 mt-1 line-clamp-2">{item.answer}</p>
              {item.user_email && (
                <p className="text-xs text-slate-500 mt-1">by {item.user_email}</p>
              )}
            </div>
            <button
              onClick={() => handleCopy(item.id)}
              className="shrink-0 px-3 py-1.5 text-xs bg-slate-700 hover:bg-teal-600 text-slate-300 hover:text-white rounded-lg transition-colors"
            >
              复制
            </button>
          </div>
        ))}
      </div>

      {totalPages > 1 && (
        <div className="flex justify-center gap-2 py-3">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            className="px-3 py-1 text-xs bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded"
          >
            ‹
          </button>
          <span className="text-xs text-slate-400 self-center">{page} / {totalPages}</span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="px-3 py-1 text-xs bg-slate-700 hover:bg-slate-600 disabled:opacity-40 rounded"
          >
            ›
          </button>
        </div>
      )}
    </div>
  )
}
