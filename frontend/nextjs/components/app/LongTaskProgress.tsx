'use client'

interface Props {
  content: string
  streaming: boolean
}

interface TaskState {
  phase: 'submitted' | 'running' | 'completed' | 'failed'
  taskId: string
  progress: number
  stepLabel: string
  errorMessage?: string
  reportLinks?: { label: string; url: string }[]
  message: string
}

function parseTaskContent(content: string): TaskState | null {
  if (!content) return null

  // Detect long task content: starts with 🔬
  if (!content.includes('🔬')) return null

  // Extract task ID
  const idMatch = content.match(/任务ID:\s*(lt_\w+)/)
  const taskId = idMatch ? idMatch[1] : ''

  // Extract progress percentage
  const pctMatch = content.match(/\[(\d+)%\]/)
  const progress = pctMatch ? parseInt(pctMatch[1], 10) : 0

  // Extract step label
  const labelMatch = content.match(/\]\s*(.+?)(?:\.{2,})?$/)
  let stepLabel = labelMatch ? labelMatch[1].trim() : ''

  // Completed state
  if (content.includes('✅')) {
    const linkRegex = /\[(DOCX|PDF)\]\(([^)]+)\)/g
    const reportLinks: { label: string; url: string }[] = []
    let m
    while ((m = linkRegex.exec(content)) !== null) {
      reportLinks.push({ label: m[1], url: m[2] })
    }
    return {
      phase: 'completed',
      taskId,
      progress: 100,
      stepLabel: '',
      reportLinks,
      message: content.split('\n').find(l => l.startsWith('✅')) || '',
    }
  }

  // Failed state
  if (content.includes('❌')) {
    return {
      phase: 'failed',
      taskId,
      progress,
      stepLabel,
      errorMessage: content.replace(/^.*?❌\s*/, '').trim(),
      message: content,
    }
  }

  // Submitted / Running
  if (taskId && progress === 0 && !stepLabel) {
    return { phase: 'submitted', taskId, progress: 0, stepLabel: '', message: content }
  }

  if (taskId && progress >= 0) {
    return { phase: 'running', taskId, progress, stepLabel, message: content }
  }

  return null
}

const PHASES = [
  { key: 'searching_patents', label: '检索', icon: '🔍' },
  { key: 'generating_columns', label: '分析框架', icon: '📊' },
  { key: 'analyzing', label: '专利分析', icon: '📄' },
  { key: 'generating_report', label: '报告撰写', icon: '📝' },
  { key: 'exporting', label: '文件导出', icon: '📦' },
]

export default function LongTaskProgress({ content, streaming }: Props) {
  const state = parseTaskContent(content)
  if (!state) return null

  return (
    <div className="lt-progress-card">
      {/* Header */}
      <div className="lt-progress-header">
        <div className="lt-progress-pulse" data-active={state.phase === 'running' || state.phase === 'submitted'} />
        <span className="lt-progress-title">
          {state.phase === 'completed'
            ? '深度分析完成'
            : state.phase === 'failed'
            ? '分析失败'
            : state.phase === 'submitted'
            ? '深度分析已提交'
            : '深度分析进行中'}
        </span>
        {state.taskId && (
          <span className="lt-progress-id">{state.taskId}</span>
        )}
      </div>

      {/* Progress bar (only in running phase) */}
      {(state.phase === 'running' || state.phase === 'submitted') && (
        <div className="lt-progress-bar-wrap">
          <div className="lt-progress-bar-track">
            <div
              className="lt-progress-bar-fill"
              style={{ width: `${Math.max(state.progress, 2)}%` }}
            />
          </div>
          <span className="lt-progress-pct">{state.progress}%</span>
        </div>
      )}

      {/* Phase indicators */}
      {(state.phase === 'running' || state.phase === 'submitted') && (
        <div className="lt-phases">
          {PHASES.map((p) => {
            // Determine which phases are active/completed based on progress
            let status: 'done' | 'active' | 'pending' = 'pending'
            if (p.key === 'searching_patents' && state.progress >= 2) status = 'done'
            else if (p.key === 'generating_columns' && state.progress >= 5) status = 'done'
            else if (p.key === 'analyzing' && state.progress >= 10) status = state.progress < 75 ? 'active' : 'done'
            else if (p.key === 'generating_report' && state.progress >= 80) status = state.progress < 90 ? 'active' : 'done'
            else if (p.key === 'exporting' && state.progress >= 92) status = 'active'

            // Highlight current based on step label
            if (state.stepLabel.includes(p.label) && status !== 'done') status = 'active'

            return (
              <div
                key={p.key}
                className={`lt-phase-dot ${status}`}
                title={p.label}
              >
                <span className="lt-phase-icon">{p.icon}</span>
                <span className="lt-phase-label">{p.label}</span>
              </div>
            )
          })}
        </div>
      )}

      {/* Current step */}
      {state.stepLabel && state.phase === 'running' && (
        <p className="lt-current-step">{state.stepLabel}</p>
      )}

      {/* Completed: download buttons */}
      {state.phase === 'completed' && state.reportLinks && (
        <div className="lt-downloads">
          {state.reportLinks.map((link) => (
            <a
              key={link.label}
              href={link.url}
              className={`lt-dl-btn ${link.label.toLowerCase()}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              <span className="lt-dl-icon">
                {link.label === 'PDF' ? '📕' : '📘'}
              </span>
              <span className="lt-dl-label">
                下载 {link.label}
              </span>
              <svg className="lt-dl-arrow" viewBox="0 0 16 16" fill="none" strokeWidth="2" strokeLinecap="round">
                <path d="M8 3v8M4 8l4 4 4-4" />
              </svg>
            </a>
          ))}
        </div>
      )}

      {/* Failed: error */}
      {state.phase === 'failed' && state.errorMessage && (
        <p className="lt-error">{state.errorMessage}</p>
      )}

      {/* Submitted: waiting message */}
      {state.phase === 'submitted' && (
        <p className="lt-current-step">正在后台执行分析任务，您可以继续对话...</p>
      )}
    </div>
  )
}
