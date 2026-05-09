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
  const [leftOpen, setLeftOpen] = useState(true)

  async function loadTools() {
    try {
      const res = await queryTools({})
      setTools((res.tools || res.items || []).filter((t) => t.push === 2))
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
    <div className="flex h-full">
      {/* Left panel */}
      <div
        className={`border-r border-slate-800 flex flex-col transition-all duration-200 ${
          leftOpen ? 'w-56' : 'w-10'
        }`}
      >
        <div className="flex items-center justify-between px-3 py-3 border-b border-slate-800">
          {leftOpen && (
            <span className="text-xs font-semibold text-slate-300">已创建的 APIs</span>
          )}
          <button
            onClick={() => setLeftOpen(!leftOpen)}
            className="text-slate-400 hover:text-white text-xs ml-auto"
            title={leftOpen ? '收起' : '展开'}
          >
            {leftOpen ? '◀' : '▶'}
          </button>
        </div>

        {leftOpen && (
          <div className="flex-1 overflow-y-auto px-2 py-2 flex flex-col gap-1">
            {tools.length === 0 ? (
              <p className="text-xs text-slate-500 text-center mt-4">暂无 API</p>
            ) : (
              tools.map((t) => (
                <div
                  key={t.id}
                  className="bg-slate-800 rounded-lg px-3 py-2 flex items-start justify-between gap-1"
                >
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-slate-200 truncate">{t.title}</p>
                    <p className="text-xs text-slate-500 truncate">{t.url}</p>
                  </div>
                  <button
                    onClick={() => handleDelete(t.id)}
                    className="text-slate-500 hover:text-red-400 text-xs shrink-0"
                  >
                    ✕
                  </button>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Right panel */}
      <div className="flex-1 flex flex-col p-4 gap-3">
        <div>
          <h2 className="text-sm font-semibold text-slate-200">粘贴 OpenAPI 规范内容</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            直接粘贴自定义 JSON 或 OpenAPI/Swagger 规范
          </p>
        </div>

        <textarea
          value={raw}
          onChange={(e) => setRaw(e.target.value)}
          placeholder={PLACEHOLDER}
          rows={12}
          className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-300 font-mono placeholder-slate-600 resize-none focus:outline-none focus:border-teal-500"
        />

        {error && <p className="text-red-400 text-xs">{error}</p>}
        {success && <p className="text-teal-400 text-xs">{success}</p>}

        <div className="flex justify-end">
          <button
            onClick={handleImport}
            disabled={!raw.trim() || importing}
            className="px-5 py-2 bg-teal-600 hover:bg-teal-500 disabled:opacity-40 text-sm text-white rounded-lg"
          >
            {importing ? '导入中...' : '导入'}
          </button>
        </div>
      </div>
    </div>
  )
}
