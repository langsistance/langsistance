import { useState, useEffect } from 'react'
import { queryTools, queryToolById, createToolFromCustom, updateTool, deleteTool } from '../services/api'

const PLACEHOLDER = `{
  "url": "https://api.example.com/data",
  "method": "GET",
  "query": {},
  "header": { "x-api-key": "xxx" },
  "body": {}
}`

const METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
const CONTENT_TYPES = ['application/json', 'application/x-www-form-urlencoded', 'multipart/form-data', 'text/plain']

function getToolMethod(tool) {
  if (tool.method) return tool.method
  if (tool.params) {
    try { return JSON.parse(tool.params).method || 'GET' } catch {}
  }
  return 'GET'
}

function getToolContentType(tool) {
  if (tool.contentType) return tool.contentType
  if (tool.params) {
    try {
      const p = JSON.parse(tool.params)
      return p['Content-Type'] || p.contentType || 'application/json'
    } catch {}
  }
  return 'application/json'
}

function formatToolBody(tool) {
  if (tool.params) {
    try {
      const p = JSON.parse(tool.params)
      const d = { ...p }
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

function ToolModal({ toolId, onClose, onSaved, onDeleted }) {
  const [tool, setTool] = useState(null)
  const [form, setForm] = useState(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    queryToolById(toolId)
      .then((res) => {
        const t = res.data || res
        setTool(t)
        setForm({
          title: t.title || '',
          description: t.description || '',
          url: t.url || '',
          method: getToolMethod(t),
          contentType: getToolContentType(t),
          body: formatToolBody(t),
        })
      })
      .catch(() => setError('加载工具详情失败'))
      .finally(() => setLoading(false))
  }, [toolId])

  function set(key, val) {
    setForm((f) => ({ ...f, [key]: val }))
  }

  async function handleSave(e) {
    e.preventDefault()
    setError('')
    if (!form.title.trim()) { setError('工具名称不能为空'); return }
    if (!form.url.trim()) { setError('工具 URL 不能为空'); return }
    if (!form.body.trim()) { setError('请求体不能为空'); return }

    let parsedBody = null
    try { parsedBody = JSON.parse(form.body) } catch {}

    const params = { method: form.method, 'Content-Type': form.contentType }
    if (parsedBody && typeof parsedBody === 'object' && Object.keys(parsedBody).length > 0) {
      Object.assign(params, parsedBody)
    }

    setSaving(true)
    try {
      await updateTool({ toolId: tool.id, title: form.title, description: form.description, url: form.url, params: JSON.stringify(params) })
      onSaved()
    } catch (e) {
      setError('保存失败：' + e.message)
    } finally {
      setSaving(false)
    }
  }

  async function handleDelete() {
    if (!confirm('确认删除该 API？')) return
    try {
      await deleteTool({ toolId: tool.id })
      onDeleted()
    } catch (e) {
      setError('删除失败：' + e.message)
    }
  }

  return (
    <div className="modal">
      <div className="modal-overlay" onClick={onClose} />
      <div className="modal-content" style={{ maxWidth: 800 }}>
        <div className="modal-header">
          <h2>编辑工具</h2>
          <button className="modal-close-btn" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {loading ? (
          <div className="modal-body" style={{ padding: 32, textAlign: 'center', color: 'var(--color-text-secondary)' }}>加载中...</div>
        ) : !form ? (
          <div className="modal-body" style={{ padding: 32, textAlign: 'center', color: '#D32F2F' }}>{error || '加载失败'}</div>
        ) : (
          <form onSubmit={handleSave} style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0, overflow: 'hidden' }}>
            <div className="modal-body" style={{ flex: 1, overflowY: 'auto' }}>
              <div className="form-group">
                <label>工具名称</label>
                <input className="form-input" value={form.title} onChange={(e) => set('title', e.target.value)} required />
              </div>
              <div className="form-group">
                <label>描述</label>
                <textarea className="form-textarea" rows={2} value={form.description} onChange={(e) => set('description', e.target.value)} />
              </div>
              <div className="form-group">
                <label>API URL</label>
                <input className="form-input" value={form.url} onChange={(e) => set('url', e.target.value)} required />
              </div>
              <div className="form-group">
                <label>请求方式</label>
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
                <label>请求体</label>
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
              <button type="button" className="btn btn-secondary" style={{ color: '#D32F2F' }} onClick={handleDelete}>删除工具</button>
              <button type="submit" className="btn btn-primary" disabled={saving}>{saving ? '保存中...' : '保存工具'}</button>
              <button type="button" className="btn btn-secondary" onClick={onClose}>关闭</button>
            </div>
          </form>
        )}
      </div>
    </div>
  )
}

export default function DevTools() {
  const [tools, setTools] = useState([])
  const [raw, setRaw] = useState('')
  const [importing, setImporting] = useState(false)
  const [importError, setImportError] = useState('')
  const [importSuccess, setImportSuccess] = useState('')
  const [modal, setModal] = useState(null)

  async function loadTools() {
    try {
      const res = await queryTools({})
      setTools((res.data || []).filter((t) => t.push === 2))
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
      setImportError('JSON 格式错误，请检查输入')
      return
    }
    setImporting(true)
    try {
      await createToolFromCustom({ ...parsed, push: 2 })
      setImportSuccess('导入成功')
      setRaw('')
      await loadTools()
    } catch (e) {
      setImportError('导入失败：' + e.message)
    } finally {
      setImporting(false)
    }
  }

  return (
    <div className="page active">
      <div className="page-header">
        <h1>开发者工具</h1>
        <p>导入自定义 API 工具，与知识库结合使用</p>
      </div>

      <div className="page-content" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div className="browser-capture-layout" style={{ flex: 1, minHeight: 0 }}>
          <div className="browser-capture-sidebar">
            <div className="sidebar-section" style={{ flex: 1 }}>
              <div className="sidebar-header">
                <h3>已创建的 APIs</h3>
                <span className="api-count">{tools.length}</span>
              </div>
              <div className="apis-sidebar-list">
                {tools.length === 0 ? (
                  <div className="empty-state-small"><p>暂无 API</p></div>
                ) : (
                  tools.map((t) => (
                    <button key={t.id} className="api-sidebar-item" onClick={() => setModal(t.id)}>
                      <div className="api-sidebar-item-name">
                        <span className={`api-sidebar-item-method ${getToolMethod(t)}`}>
                          {getToolMethod(t)}
                        </span>
                        {t.title}
                      </div>
                      <div className="api-sidebar-item-path">{t.url?.replace(/^https?:\/\/[^/]+/, '') || ''}</div>
                    </button>
                  ))
                )}
              </div>
            </div>
          </div>

          <div className="browser-capture-main">
            <div className="import-method-card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <h3>粘贴 OpenAPI 规范内容</h3>
              <p style={{ color: 'var(--color-text-secondary)', fontSize: 13, marginBottom: 16 }}>
                直接粘贴自定义 JSON 或 OpenAPI/Swagger 规范
              </p>
              <textarea
                className="form-textarea"
                value={raw}
                onChange={(e) => setRaw(e.target.value)}
                placeholder={PLACEHOLDER}
                rows={12}
                style={{ flex: 1, fontFamily: 'Monaco, Courier New, monospace', fontSize: 13 }}
              />
              {importError && <p style={{ color: '#D32F2F', fontSize: 14, marginTop: 8 }}>{importError}</p>}
              {importSuccess && <p style={{ color: '#388E3C', fontSize: 14, marginTop: 8 }}>{importSuccess}</p>}
              <button
                className="btn btn-primary"
                style={{ marginTop: 10 }}
                onClick={handleImport}
                disabled={!raw.trim() || importing}
              >
                {importing ? '导入中...' : '导入'}
              </button>
            </div>
          </div>
        </div>
      </div>

      {modal && (
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
