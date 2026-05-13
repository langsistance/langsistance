'use client'

import { useState, useEffect } from 'react'
import { queryTools, createToolFromCustom, deleteTool } from '@/services/api'
import { useI18n } from '@/lib/app-i18n'

interface Tool {
  id: number
  title: string
  push?: number
  method?: string
  url?: string
}

const PLACEHOLDER = `{
  "url": "https://api.example.com/data",
  "method": "GET",
  "query": {},
  "header": { "x-api-key": "xxx" },
  "body": {}
}`

export default function DevTools() {
  const { t } = useI18n()
  const [tools, setTools] = useState<Tool[]>([])
  const [raw, setRaw] = useState('')
  const [importing, setImporting] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  async function loadTools() {
    try {
      const res = await queryTools({})
      setTools((res.data || []).filter((tool: Tool) => tool.push === 2))
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
      setError(t('alerts.openApiCheckTips'))
      return
    }
    setImporting(true)
    try {
      await createToolFromCustom({ ...parsed, push: 2 })
      setSuccess(t('notifications.dataImported'))
      setRaw('')
      await loadTools()
    } catch (e) {
      setError(t('alerts.importOpenApiFailure') + ': ' + (e as Error).message)
    } finally {
      setImporting(false)
    }
  }

  async function handleDelete(id: number) {
    if (!confirm(t('confirmations.deleteTool'))) return
    await deleteTool({ id })
    loadTools()
  }

  return (
    <div className="page active">
      <div className="page-header">
        <h1>{t('browser.devTools')}</h1>
        <p>{t('developer.description')}</p>
      </div>

      <div className="page-content">
        <div className="browser-capture-layout" style={{ height: 'auto', minHeight: 500 }}>
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
                    <button key={tool.id} className="api-sidebar-item" onClick={() => handleDelete(tool.id)}>
                      <div className="api-sidebar-item-name">
                        <span className={`api-sidebar-item-method ${tool.method || 'GET'}`}>
                          {tool.method || 'GET'}
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
            <div className="import-method-card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
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
              {error && (
                <p style={{ color: '#D32F2F', fontSize: 14, marginTop: 8 }}>{error}</p>
              )}
              {success && (
                <p style={{ color: '#388E3C', fontSize: 14, marginTop: 8 }}>{success}</p>
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
    </div>
  )
}
