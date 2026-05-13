import { useState, useEffect } from 'react'
import { queryTools, createToolFromCustom, deleteTool } from '../services/api'

const PLACEHOLDER = `{
  "url": "https://api.example.com/data",
  "method": "GET",
  "query": {},
  "header": { "x-api-key": "xxx" },
  "body": {}
}`

export default function DevTools() {
  const [tools, setTools] = useState([])
  const [raw, setRaw] = useState('')
  const [importing, setImporting] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  async function loadTools() {
    try {
      const res = await queryTools({})
      setTools((res.data || []).filter((t) => t.push === 2))
    } catch (e) { console.error(e) }
  }

  useEffect(() => { loadTools() }, [])

  async function handleImport() {
    setError('')
    setSuccess('')
    let parsed
    try {
      parsed = JSON.parse(raw)
    } catch {
      setError('JSON 格式错误，请检查输入')
      return
    }
    setImporting(true)
    try {
      await createToolFromCustom({ ...parsed, push: 2 })
      setSuccess('导入成功')
      setRaw('')
      await loadTools()
    } catch (e) {
      setError('导入失败：' + e.message)
    } finally {
      setImporting(false)
    }
  }

  async function handleDelete(id) {
    if (!confirm('确认删除该 API？')) return
    await deleteTool({ id })
    loadTools()
  }

  return (
    <div className="page active">
      <div className="page-header">
        <h1>开发者工具</h1>
        <p>导入自定义 API 工具，与知识库结合使用</p>
      </div>

      <div className="page-content" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div className="browser-capture-layout" style={{ flex: 1, minHeight: 0 }}>
          {/* Left panel: API list */}
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
                    <button key={t.id} className="api-sidebar-item" onClick={() => handleDelete(t.id)}>
                      <div className="api-sidebar-item-name">
                        <span className={`api-sidebar-item-method ${t.method || 'GET'}`}>
                          {t.method || 'GET'}
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

          {/* Right panel: Paste raw */}
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
              {error && <p style={{ color: '#D32F2F', fontSize: 14, marginTop: 8 }}>{error}</p>}
              {success && <p style={{ color: '#388E3C', fontSize: 14, marginTop: 8 }}>{success}</p>}
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
    </div>
  )
}
