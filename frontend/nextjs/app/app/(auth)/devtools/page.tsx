'use client'

import { useState, useEffect } from 'react'
import { queryTools, queryToolById, createToolFromCustom, updateTool, deleteTool } from '@/services/api'
import { useI18n } from '@/lib/app-i18n'

interface Tool {
  id: number
  title: string
  push?: number
  method?: string
  url?: string
  description?: string
  params?: string
  body?: string
  contentType?: string
}

interface ToolForm {
  title: string
  description: string
  url: string
  method: string
  contentType: string
  body: string
}

const PLACEHOLDER = `{
  "url": "https://api.example.com/data",
  "method": "GET",
  "query": {},
  "header": { "x-api-key": "xxx" },
  "body": {}
}`

const METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
const CONTENT_TYPES = ['application/json', 'application/x-www-form-urlencoded', 'multipart/form-data', 'text/plain']

function getToolMethod(tool: Tool): string {
  if (tool.method) return tool.method
  if (tool.params) {
    try { return JSON.parse(tool.params).method || 'GET' } catch {}
  }
  return 'GET'
}

function getToolContentType(tool: Tool): string {
  if (tool.contentType) return tool.contentType
  if (tool.params) {
    try {
      const p = JSON.parse(tool.params)
      return p['Content-Type'] || p.contentType || 'application/json'
    } catch {}
  }
  return 'application/json'
}

function formatToolBody(tool: Tool): string {
  if (tool.params) {
    try {
      const p = JSON.parse(tool.params)
      const d: Record<string, unknown> = { ...p }
      delete d.method
      delete d['Content-Type']
      delete d.contentType
      return JSON.stringify(d, null, 2)
    } catch {}
    return tool.params
  }
  if (tool.body) {
    if (tool.contentType === 'application/json') {
      try { return JSON.stringify(JSON.parse(tool.body), null, 2) } catch {}
    }
    return tool.body
  }
  return '{}'
}

function ToolModal({ toolId, onClose, onSaved, onDeleted }: {
  toolId: number
  onClose: () => void
  onSaved: () => void
  onDeleted: () => void
}) {
  const { t } = useI18n()
  const [tool, setTool] = useState<Tool | null>(null)
  const [form, setForm] = useState<ToolForm | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    queryToolById(toolId)
      .then((res) => {
        const tl: Tool = res.data || res
        setTool(tl)
        setForm({
          title: tl.title || '',
          description: tl.description || '',
          url: tl.url || '',
          method: getToolMethod(tl),
          contentType: getToolContentType(tl),
          body: formatToolBody(tl),
        })
      })
      .catch(() => setError(t('tools.noTools')))
      .finally(() => setLoading(false))
  }, [toolId, t])

  function set(key: keyof ToolForm, val: string) {
    setForm((f) => f ? { ...f, [key]: val } : f)
  }

  async function handleSave(e: React.FormEvent) {
    e.preventDefault()
    if (!form || !tool) return
    setError('')
    if (!form.title.trim()) { setError(t('tools.name') + ' is required'); return }
    if (!form.url.trim()) { setError(t('tools.url') + ' is required'); return }

    let parsedBody: Record<string, unknown> | null = null
    try { parsedBody = JSON.parse(form.body) } catch {}

    const params: Record<string, unknown> = { method: form.method, 'Content-Type': form.contentType }
    if (parsedBody && typeof parsedBody === 'object' && Object.keys(parsedBody).length > 0) {
      Object.assign(params, parsedBody)
    }

    setSaving(true)
    try {
      await updateTool({ toolId: tool.id, title: form.title, description: form.description, url: form.url, params: JSON.stringify(params) })
      onSaved()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSaving(false)
    }
  }

  async function handleDelete() {
    if (!tool || !confirm(t('confirmations.deleteTool'))) return
    try {
      await deleteTool({ toolId: tool.id })
      onDeleted()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  return (
    <div className="modal">
      <div className="modal-overlay" onClick={onClose} />
      <div className="modal-content" style={{ maxWidth: 800 }}>
        <div className="modal-header">
          <h2>{t('tools.edit')}</h2>
          <button className="modal-close-btn" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {loading ? (
          <div className="modal-body" style={{ padding: 32, textAlign: 'center', color: 'var(--color-text-secondary)' }}>{t('common.loading')}</div>
        ) : !form ? (
          <div className="modal-body" style={{ padding: 32, textAlign: 'center', color: '#D32F2F' }}>{error || t('tools.noTools')}</div>
        ) : (
          <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
            <div className="modal-body" style={{ flex: 1, overflowY: 'auto' }}>
              <div className="form-group">
                <label>{t('tools.name')}</label>
                <input className="form-input" value={form.title} onChange={(e) => set('title', e.target.value)} required />
              </div>
              <div className="form-group">
                <label>{t('tools.description')}</label>
                <textarea className="form-textarea" rows={2} value={form.description} onChange={(e) => set('description', e.target.value)} />
              </div>
              <div className="form-group">
                <label>{t('tools.url')}</label>
                <input className="form-input" value={form.url} onChange={(e) => set('url', e.target.value)} required />
              </div>
              <div className="form-group">
                <label>{t('tools.method')}</label>
                <select className="form-input" value={form.method} onChange={(e) => set('method', e.target.value)}>
                  {METHODS.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Content-Type</label>
                <select className="form-input" value={form.contentType} onChange={(e) => set('contentType', e.target.value)}>
                  {CONTENT_TYPES.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>{t('developer.pasteOpenApi')}</label>
                <textarea
                  className="form-textarea"
                  rows={5}
                  value={form.body}
                  onChange={(e) => set('body', e.target.value)}
                  style={{ fontFamily: 'Monaco, Courier New, monospace', fontSize: 12 }}
                />
              </div>
              {error && <p style={{ color: '#D32F2F', fontSize: 14, marginTop: -8 }}>{error}</p>}
            </div>
            <div className="modal-footer">
              <button type="button" className="btn btn-secondary" style={{ color: '#D32F2F' }} onClick={handleDelete}>{t('tools.delete')}</button>
              <button type="submit" className="btn btn-primary" disabled={saving}>{saving ? t('common.loading') : t('tools.save')}</button>
              <button type="button" className="btn btn-secondary" onClick={onClose}>{t('common.close')}</button>
            </div>
          </form>
        )}
      </div>
    </div>
  )
}

export default function DevTools() {
  const { t } = useI18n()
  const [tools, setTools] = useState<Tool[]>([])
  const [raw, setRaw] = useState('')
  const [importing, setImporting] = useState(false)
  const [importError, setImportError] = useState('')
  const [importSuccess, setImportSuccess] = useState('')
  const [modal, setModal] = useState<number | null>(null)

  async function loadTools() {
    try {
      const res = await queryTools({})
      setTools((res.data || []).filter((tool: Tool) => tool.push === 2))
    } catch (e) { console.error(e) }
  }

  useEffect(() => { loadTools() }, [])

  async function handleImport() {
    setImportError('')
    setImportSuccess('')
    let parsed
    try {
      parsed = JSON.parse(raw)
    } catch {
      setImportError(t('alerts.openApiCheckTips'))
      return
    }
    setImporting(true)
    try {
      await createToolFromCustom({ ...parsed, push: 2 })
      setImportSuccess(t('notifications.dataImported'))
      setRaw('')
      await loadTools()
    } catch (e) {
      setImportError(t('alerts.importOpenApiFailure') + ': ' + (e as Error).message)
    } finally {
      setImporting(false)
    }
  }

  return (
    <div className="page active">
      <div className="page-header">
        <h1>{t('browser.devTools')}</h1>
        <p>{t('developer.description')}</p>
      </div>

      <div className="page-content" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div className="browser-capture-layout" style={{ flex: 1, minHeight: 0 }}>
          <div className="browser-capture-sidebar">
            <div className="sidebar-section" style={{ flex: 1 }}>
              <div className="sidebar-header">
                <h3>{t('developer.apiCreated')}</h3>
                <span className="api-count">{tools.length}</span>
              </div>
              <div className="apis-sidebar-list">
                {tools.length === 0 ? (
                  <div className="empty-state-small"><p>{t('developer.noApi')}</p></div>
                ) : (
                  tools.map((tool) => (
                    <button key={tool.id} className="api-sidebar-item" onClick={() => setModal(tool.id)}>
                      <div className="api-sidebar-item-name">
                        <span className={`api-sidebar-item-method ${getToolMethod(tool)}`}>
                          {getToolMethod(tool)}
                        </span>
                        {tool.title}
                      </div>
                      <div className="api-sidebar-item-path">{tool.url?.replace(/^https?:\/\/[^/]+/, '') || ''}</div>
                    </button>
                  ))
                )}
              </div>
            </div>
          </div>

          <div className="browser-capture-main">
            <div className="import-method-card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <h3>{t('developer.pasteOpenApi')}</h3>
              <p style={{ color: 'var(--color-text-secondary)', fontSize: 13, marginBottom: 16 }}>
                {t('developer.pasteOpenApiDesc')}
              </p>
              <textarea
                className="form-textarea"
                value={raw}
                onChange={(e) => setRaw(e.target.value)}
                placeholder={PLACEHOLDER}
                rows={12}
                style={{ flex: 1, fontFamily: 'Monaco, Courier New, monospace', fontSize: 13 }}
              />
              {importError && (
                <p style={{ color: '#D32F2F', fontSize: 14, marginTop: 8 }}>{importError}</p>
              )}
              {importSuccess && (
                <p style={{ color: '#388E3C', fontSize: 14, marginTop: 8 }}>{importSuccess}</p>
              )}
              <button
                className="btn btn-primary"
                style={{ marginTop: 10 }}
                onClick={handleImport}
                disabled={!raw.trim() || importing}
              >
                {importing ? t('common.loading') : t('common.import')}
              </button>
            </div>
          </div>
        </div>
      </div>

      {modal !== null && (
        <ToolModal
          toolId={modal}
          onClose={() => setModal(null)}
          onSaved={() => { setModal(null); loadTools() }}
          onDeleted={() => { setModal(null); loadTools() }}
        />
      )}
    </div>
  )
}
