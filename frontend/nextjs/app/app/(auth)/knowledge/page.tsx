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
import { filterKnowledgeBaseTools } from '@/lib/toolFilters'
import { KNOWLEDGE_LIST_PAGE_SIZE } from '@/lib/appUiConfig'
import { getKnowledgeTypeBadge } from '@/lib/knowledgeTypeBadge'
import Pagination from '@/components/app/Pagination'

interface KnowledgeItem {
  id?: number
  question: string
  answer: string
  description: string
  tool_id: number | string
  public: number
  type?: number
  model_name?: string
  params?: string
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

function KnowledgeModal({ item, tools, knowledgeOptions, onClose, onSave, onDelete }: {
  item: KnowledgeItem | null
  tools: Tool[]
  knowledgeOptions: KnowledgeItem[]
  onClose: () => void
  onSave: (form: KnowledgeItem) => Promise<void>
  onDelete: (id: number) => Promise<void>
}) {
  const { t, lang } = useI18n()
  const [form, setForm] = useState<KnowledgeItem>(
    item
      ? { ...item }
      : { question: '', answer: '', description: '', tool_id: '', public: 1, type: 1 }
  )
  const [saveError, setSaveError] = useState('')
  const initialStepIds = (() => {
    try {
      const parsed = item?.params ? JSON.parse(item.params) : null
      if (parsed?.type === 'workflow' && Array.isArray(parsed.steps)) {
        return parsed.steps.map((step: { knowledge_id?: number }) => String(step.knowledge_id || '')).filter(Boolean)
      }
    } catch {
      return []
    }
    return []
  })()
  const [stepIds, setStepIds] = useState<string[]>(initialStepIds.length ? initialStepIds : ['', ''])
  const isWorkflow = Number(form.type || 1) === 2

  // Hide tool selector when editing and associated tool is unavailable in the filtered list.
  const hasValidTool = item?.tool_id
    ? tools.some((t) => String(t.id) === String(item.tool_id))
    : true
  const showToolSelect = hasValidTool

  function set(key: string, val: unknown) {
    setForm((f) => ({ ...f, [key]: val }))
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setSaveError('')
    try {
      const nextForm = { ...form }
      if (isWorkflow) {
        const selectedSteps = stepIds.filter(Boolean)
        if (!nextForm.question.trim()) {
          throw new Error(lang === 'en' ? 'Name is required' : '请输入组合知识名称')
        }
        if (selectedSteps.length < 2) {
          throw new Error(lang === 'en' ? 'Select at least two knowledge steps' : '请至少选择两个步骤知识')
        }
        nextForm.tool_id = 0
        nextForm.answer = `组合知识：按顺序执行 ${selectedSteps.length} 个知识步骤。`
        nextForm.params = JSON.stringify({
          type: 'workflow',
          version: 1,
          mode: 'context_chain',
          steps: selectedSteps.map((knowledgeId, index) => ({
            id: `step_${index + 1}`,
            knowledge_id: Number(knowledgeId),
          })),
        })
      }
      await onSave(nextForm)
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
        <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
          <div className="modal-body">
            <div className="form-group">
              <label>{lang === 'en' ? 'Knowledge Type' : '知识类型'}</label>
              <select
                className="form-input"
                value={String(form.type || 1)}
                onChange={(e) => set('type', Number(e.target.value))}
              >
                <option value="1">{lang === 'en' ? 'Normal Knowledge' : '普通知识'}</option>
                <option value="2">{lang === 'en' ? 'Composed Knowledge' : '组合知识'}</option>
              </select>
            </div>
            {!isWorkflow && showToolSelect && (
              <div className="form-group">
                <label>{t('modals.knowledgeCreate.selectTool')}</label>
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
            )}
            <h4>{t('modals.knowledgeDetails.knowledgeContent')}</h4>
            <div className="form-section">
              <div className="form-group">
                <label>{isWorkflow ? (lang === 'en' ? 'Name' : '名称') : t('modals.knowledgeCreate.question')}</label>
                <input
                  id="knowledgeQuestion"
                  className="form-input"
                  placeholder={isWorkflow ? (lang === 'en' ? 'e.g. Find all patent documents by publication ID' : '例如：根据专利公开 ID 查询所有文档') : t('modals.knowledgeCreate.questionPlaceholder')}
                  value={form.question}
                  onChange={(e) => set('question', e.target.value)}
                  required
                />
              </div>
              {isWorkflow ? (
                <div className="form-group">
                  <label>{lang === 'en' ? 'Steps' : '步骤知识'}</label>
                  <div className="workflow-step-list">
                    {stepIds.map((stepId, index) => (
                      <div className="workflow-step-row" key={index}>
                        <span className="workflow-step-index">{index + 1}</span>
                        <select
                          className="form-input"
                          value={stepId}
                          onChange={(e) => {
                            const next = [...stepIds]
                            next[index] = e.target.value
                            setStepIds(next)
                          }}
                        >
                          <option value="">{lang === 'en' ? 'Select knowledge' : '选择知识'}</option>
                          {knowledgeOptions
                            .filter((option) => option.id !== item?.id && Number(option.type || 1) === 1)
                            .map((option) => (
                              <option key={option.id} value={option.id}>{option.question}</option>
                            ))}
                        </select>
                        {stepIds.length > 2 && (
                          <button
                            type="button"
                            className="btn btn-secondary workflow-step-remove"
                            onClick={() => setStepIds(stepIds.filter((_, i) => i !== index))}
                          >
                            {lang === 'en' ? 'Remove' : '移除'}
                          </button>
                        )}
                      </div>
                    ))}
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => setStepIds([...stepIds, ''])}
                    >
                      {lang === 'en' ? 'Add Step' : '添加步骤'}
                    </button>
                  </div>
                </div>
              ) : (
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
              )}
              <div className="form-group">
                <label>{t('modals.toolDetails.description')}</label>
                <textarea
                  className="form-textarea"
                  placeholder={t('modals.toolCreate.knowledgeDescPlaceholder')}
                  value={form.description}
                  onChange={(e) => set('description', e.target.value)}
                  rows={2}
                />
              </div>
            </div>
            <div className="form-group">
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  id="knowledgePublic"
                  checked={form.public === 2}
                  onChange={(e) => set('public', e.target.checked ? 2 : 1)}
                />
                {t('modals.knowledgeCreate.makePublic')}
              </label>
            </div>
            {item && item.update_time && (
              <div className="metadata-section">
                <h4>{t('modals.knowledgeDetails.basicInfo')}</h4>
                <div className="detail-item">
                  <strong>{t('modals.knowledgeDetails.updatedAt')}:</strong>{' '}
                  {new Date(item.update_time).toLocaleString(lang === 'en' ? 'en-US' : 'zh-CN')}
                </div>
              </div>
            )}
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
              >{t('modals.knowledgeDetails.deleteKnowledge')}</button>
            )}
            <button type="button" className="btn btn-secondary" onClick={onClose}>{t('common.cancel')}</button>
            <button type="submit" className="btn btn-primary">
              {item ? t('knowledge.update') : t('modals.knowledgeCreate.createToolKnowledge')}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function Knowledge() {
  const { t, lang } = useI18n()
  const [items, setItems] = useState<KnowledgeItem[]>([])
  const [tools, setTools] = useState<Tool[]>([])
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [modal, setModal] = useState<KnowledgeItem | 'create' | null>(null)
  const PAGE_SIZE = KNOWLEDGE_LIST_PAGE_SIZE

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
      .then((res) => setTools(filterKnowledgeBaseTools(res.data || [])))
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
        type: form.type || 1,
      })
    } else {
      await createKnowledge({
        question: form.question,
        answer: form.answer,
        description: form.description,
        public: form.public || 1,
        toolId: form.tool_id ? Number(form.tool_id) : 0,
        params: form.params || '',
        modelName: form.model_name,
        type: form.type || 1,
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
            {items.map((item) => {
              const typeBadge = getKnowledgeTypeBadge(item.type, lang, item.params)
              return (
                <div key={item.id} className="knowledge-card" onClick={() => setModal(item)}>
                  <div className="knowledge-card-header">
                    <div className="knowledge-card-title">{item.question}</div>
                    <span className={typeBadge.className}>{typeBadge.label}</span>
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
              )
            })}
          </div>
        )}

        <Pagination page={page} totalPages={totalPages} onChange={setPage} />
      </div>

      {modal && (
        <KnowledgeModal
          item={modal === 'create' ? null : modal}
          tools={tools}
          knowledgeOptions={items}
          onClose={() => setModal(null)}
          onSave={handleSave}
          onDelete={handleDelete}
        />
      )}
    </div>
  )
}
