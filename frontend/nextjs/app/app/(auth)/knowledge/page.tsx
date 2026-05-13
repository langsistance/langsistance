'use client'

import { useState, useEffect, useCallback } from 'react'
import {
  queryKnowledge,
  createKnowledge,
  updateKnowledge,
  deleteKnowledge,
  queryTools,
} from '@/services/api'
import { useI18n } from '@/lib/app-i18n'
import Pagination from '@/components/app/Pagination'

interface KnowledgeItem {
  id?: number
  question: string
  answer: string
  description: string
  tool_id: number | string
  public: number
  title?: string
  update_time?: string
}

interface Tool {
  id: number
  title: string
  push?: number
  method?: string
  url?: string
}

function KnowledgeModal({ item, tools, onClose, onSave, onDelete }: {
  item: KnowledgeItem | null
  tools: Tool[]
  onClose: () => void
  onSave: (form: KnowledgeItem) => Promise<void>
  onDelete: (id: number) => Promise<void>
}) {
  const { t, lang } = useI18n()
  const [form, setForm] = useState<KnowledgeItem>(
    item
      ? { ...item }
      : { question: '', answer: '', description: '', tool_id: '', public: 0 }
  )
  const [saveError, setSaveError] = useState('')

  function set(key: string, val: unknown) {
    setForm((f) => ({ ...f, [key]: val }))
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setSaveError('')
    try {
      await onSave(form)
      onClose()
    } catch (err) {
      setSaveError((err as Error).message || (lang === 'en' ? 'Save failed' : '保存失败'))
    }
  }

  return (
    <div className="modal">
      <div className="modal-overlay" onClick={onClose} />
      <div className="modal-content knowledge-editor-modal">
        <div className="modal-header">
          <h2>{item ? t('knowledge.edit') : t('knowledge.create')}</h2>
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
              <label>{t('modals.knowledgeCreate.question')}</label>
              <input
                id="knowledgeQuestion"
                className="form-input"
                placeholder={t('modals.knowledgeCreate.questionPlaceholder')}
                value={form.question}
                onChange={(e) => set('question', e.target.value)}
                required
              />
            </div>
            <div className="form-group">
              <label>{t('modals.knowledgeCreate.answer')}</label>
              <textarea
                id="knowledgeAnswer"
                className="form-textarea"
                placeholder={t('modals.knowledgeCreate.answerPlaceholder')}
                value={form.answer}
                onChange={(e) => set('answer', e.target.value)}
                required
                rows={4}
              />
            </div>
            <div className="form-group">
              <label>{lang === 'en' ? 'Description (optional)' : '描述（可选）'}</label>
              <textarea
                className="form-textarea"
                placeholder={lang === 'en' ? 'Additional notes' : '补充说明'}
                value={form.description}
                onChange={(e) => set('description', e.target.value)}
                rows={2}
              />
            </div>
            <div className="form-group">
              <label>{lang === 'en' ? 'Associated Tool' : '关联工具'}</label>
              <select
                id="knowledgeToolSelect"
                className="form-input"
                value={String(form.tool_id)}
                onChange={(e) => set('tool_id', e.target.value)}
              >
                <option value="">{lang === 'en' ? 'No linked tool' : '无关联工具'}</option>
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
                {t('modals.knowledgeCreate.makePublic')}
              </label>
            </div>
            {saveError && (
              <p style={{ color: '#D32F2F', fontSize: 14, marginTop: -8 }}>{saveError}</p>
            )}
          </div>
          <div className="modal-footer">
            {item && (
              <button
                type="button"
                className="btn btn-secondary"
                style={{ color: '#D32F2F', marginRight: 'auto' }}
                onClick={() => onDelete(item.id!)}
              >{t('common.delete')}</button>
            )}
            <button type="button" className="btn btn-secondary" onClick={onClose}>{t('common.cancel')}</button>
            <button type="submit" className="btn btn-primary">{t('common.save')}</button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function Knowledge() {
  const { t } = useI18n()
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
      const data = res.data
      const knowledge: KnowledgeItem[] = Array.isArray(data) ? data : (data?.knowledge || [])
      const toolsInResponse: Tool[] = Array.isArray(data) ? [] : (data?.tools || [])
      const processed = knowledge.map((item) => ({
        ...item,
        title: toolsInResponse.find((t) => t.id === item.tool_id)?.title || '',
      }))
      setItems(processed)
      setTotal(res.total || 0)
    } catch (e) {
      console.error(e)
    }
  }, [debouncedSearch, page])

  useEffect(() => { load() }, [load])

  useEffect(() => {
    queryTools({})
      .then((res) => setTools((res.data || []).filter((tool: Tool) => tool.push === 2)))
      .catch(() => {})
  }, [])

  async function handleSave(form: KnowledgeItem) {
    if (form.id) {
      await updateKnowledge({
        knowledgeId: form.id,
        question: form.question,
        description: form.description,
        answer: form.answer,
        public: form.public,
        modelName: form.model_name,
        toolId: form.tool_id ? Number(form.tool_id) : 0,
        params: form.params || '',
      })
    } else {
      await createKnowledge({
        question: form.question,
        answer: form.answer,
        description: form.description,
        public: form.public || 0,
        toolId: form.tool_id ? Number(form.tool_id) : 0,
        params: form.params || '',
        modelName: form.model_name,
      })
    }
    load()
  }

  async function handleDelete(id: number) {
    if (!confirm(t('alerts.confirmDeleteKnowledge'))) return
    await deleteKnowledge({ knowledgeId: id })
    setModal(null)
    load()
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)

  return (
    <div className="page active">
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>{t('knowledge.title')}</h1>
            <p>{t('knowledge.description')}</p>
          </div>
          <button className="btn btn-primary" onClick={() => setModal('create')}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            {t('knowledge.create')}
          </button>
        </div>
      </div>

      <div className="page-content">
        <div className="knowledge-search">
          <input
            type="text"
            className="knowledge-search-input"
            placeholder={t('knowledge.search')}
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1) }}
          />
        </div>

        {items.length === 0 ? (
          <div className="empty-state">
            <p>{t('knowledge.noKnowledge')}</p>
          </div>
        ) : (
          <div className="knowledge-list">
            {items.map((item) => (
              <div key={item.id} className="knowledge-card" onClick={() => setModal(item)}>
                <div className="knowledge-card-header">
                  <div className="knowledge-card-title">{item.question}</div>
                </div>
                <div className="knowledge-card-content">{item.answer}</div>
                {item.title && (
                  <div className="knowledge-card-apis">
                    <span className="knowledge-api-badge">{item.title}</span>
                  </div>
                )}
                <div className="knowledge-card-footer">
                  <span>📅 {item.update_time ? new Date(item.update_time).toLocaleDateString('zh-CN') : ''}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {totalPages > 1 && (
          <Pagination page={page} totalPages={totalPages} onChange={setPage} />
        )}
      </div>

      {modal && (
        <KnowledgeModal
          item={modal === 'create' ? null : modal}
          tools={tools}
          onClose={() => setModal(null)}
          onSave={handleSave}
          onDelete={handleDelete}
        />
      )}
    </div>
  )
}
