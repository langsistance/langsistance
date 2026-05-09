'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  queryKnowledge,
  createKnowledge,
  updateKnowledge,
  deleteKnowledge,
  queryTools,
} from '@/services/api'

interface KnowledgeItem {
  id?: number
  question: string
  answer: string
  description: string
  tool_id: number | string
  public: number
}

interface Tool {
  id: number
  title: string
  push?: number
  method?: string
  url?: string
}

function KnowledgeModal({
  item,
  tools,
  onClose,
  onSave,
}: {
  item: KnowledgeItem | null
  tools: Tool[]
  onClose: () => void
  onSave: (form: KnowledgeItem) => Promise<void>
}) {
  const [form, setForm] = useState<KnowledgeItem>(
    item || { question: '', answer: '', description: '', tool_id: '', public: 0 }
  )

  function set(key: string, val: unknown) {
    setForm((f) => ({ ...f, [key]: val }))
  }

  const [saveError, setSaveError] = useState('')
  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setSaveError('')
    try {
      await onSave(form)
      onClose()
    } catch (err) {
      setSaveError((err as Error).message || '保存失败')
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-slate-800 rounded-xl p-6 w-full max-w-lg border border-slate-700">
        <h3 className="text-base font-semibold mb-4">{item ? '编辑知识库' : '创建知识库'}</h3>
        <form onSubmit={submit} className="flex flex-col gap-3">
          <input
            placeholder="问题（AI 匹配时使用）"
            value={form.question}
            onChange={(e) => set('question', e.target.value)}
            required
            className="bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-teal-500"
          />
          <textarea
            placeholder="答案 / 工具调用说明"
            value={form.answer}
            onChange={(e) => set('answer', e.target.value)}
            required
            rows={3}
            className="bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 resize-none focus:outline-none focus:border-teal-500"
          />
          <textarea
            placeholder="描述（可选）"
            value={form.description}
            onChange={(e) => set('description', e.target.value)}
            rows={2}
            className="bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 resize-none focus:outline-none focus:border-teal-500"
          />
          <select
            value={form.tool_id as string | number}
            onChange={(e) => set('tool_id', e.target.value)}
            className="bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-teal-500"
          >
            <option value="">无关联工具</option>
            {tools.map((tool) => (
              <option key={tool.id} value={tool.id}>
                {tool.title}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={!!form.public}
              onChange={(e) => set('public', e.target.checked ? 1 : 0)}
              className="accent-teal-600"
            />
            公开（在社区可见）
          </label>
          {saveError && <p className="text-red-400 text-xs">{saveError}</p>}
          <div className="flex justify-end gap-2 mt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm text-slate-300 hover:text-white"
            >
              取消
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-teal-600 hover:bg-teal-500 text-sm text-white rounded-lg"
            >
              保存
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function Knowledge() {
  const [items, setItems] = useState<KnowledgeItem[]>([])
  const [tools, setTools] = useState<Tool[]>([])
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [modal, setModal] = useState<KnowledgeItem | 'create' | null>(null)
  const PAGE_SIZE = 10

  const [debouncedSearch, setDebouncedSearch] = useState(search)
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(search), 300)
    return () => clearTimeout(timer)
  }, [search])

  const load = useCallback(async () => {
    try {
      const res = await queryKnowledge({ search: debouncedSearch, page, limit: PAGE_SIZE })
      setItems(res.items || res.knowledge || [])
      setTotal(res.total || 0)
    } catch (e) {
      console.error(e)
    }
  }, [debouncedSearch, page])

  useEffect(() => { load() }, [load])
  useEffect(() => {
    queryTools({ push: 2 })
      .then((res) => setTools((res.tools || res.items || []).filter((tool: Tool) => tool.push === 2)))
      .catch(() => {})
  }, [])

  async function handleSave(form: KnowledgeItem) {
    if (form.id) {
      await updateKnowledge(form)
    } else {
      await createKnowledge(form)
    }
    load()
  }

  async function handleDelete(id: number | undefined) {
    if (!confirm('确认删除？')) return
    await deleteKnowledge({ id })
    load()
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">知识库</h2>
          <p className="text-xs text-slate-500">管理 API 文档和使用说明</p>
        </div>
        <button
          onClick={() => setModal('create')}
          className="px-3 py-1.5 bg-teal-600 hover:bg-teal-500 text-xs text-white rounded-lg"
        >
          + 创建知识库
        </button>
      </div>

      <div className="px-4 py-3">
        <input
          placeholder="搜索知识库..."
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(1) }}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-teal-500"
        />
      </div>

      <div className="flex-1 overflow-y-auto px-4 flex flex-col gap-2">
        {items.map((item) => (
          <div key={item.id} className="bg-slate-800 rounded-lg p-4">
            <div className="flex items-start justify-between gap-2">
              <p className="text-sm font-medium text-slate-200 flex-1">{item.question}</p>
              <div className="flex gap-2 shrink-0">
                <button
                  onClick={() => setModal(item)}
                  className="text-xs text-slate-400 hover:text-white"
                >
                  编辑
                </button>
                <button
                  onClick={() => handleDelete(item.id)}
                  className="text-xs text-slate-400 hover:text-red-400"
                >
                  删除
                </button>
              </div>
            </div>
            <p className="text-xs text-slate-400 mt-1 line-clamp-2">{item.answer}</p>
            {(item.tool_id as number) > 0 && (
              <span className="inline-block mt-2 text-xs text-teal-400 bg-teal-900/30 px-2 py-0.5 rounded">
                {tools.find((tool) => tool.id === item.tool_id)?.title || `Tool #${item.tool_id}`}
              </span>
            )}
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
