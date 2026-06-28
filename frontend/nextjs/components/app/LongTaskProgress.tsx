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

function hasProgressMarker(content: string): boolean {
  return /\[\d+%\]/.test(content)
}

function parseTaskContent(content: string): TaskState | null {
  if (!content) return null

  // Detect long task content — explicit markers OR progress percentage pattern
  const isLongTask =
    content.includes('🔬') ||
    content.includes('✅') ||
    content.includes('❌') ||
    hasProgressMarker(content)
  if (!isLongTask) return null

  // Extract task ID from text or URL
  let taskId = ''
  const idMatch = content.match(/任务ID:\s*(lt_\w+)/)
  if (idMatch) {
    taskId = idMatch[1]
  } else {
    const urlMatch = content.match(/long_task\/(lt_\w+)/)
    if (urlMatch) taskId = urlMatch[1]
  }

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

  // Submitted: explicit task ID + "提交" keyword, no progress
  if (taskId && !stepLabel && !content.includes('✅') && !content.includes('❌') && (content.includes('已提交') || content.includes('submitted'))) {
    return { phase: 'submitted', taskId, progress: 0, stepLabel: '', message: content }
  }

  // Running: has 🔬 marker or progress percentage (taskId may be absent from polling updates)
  if (content.includes('🔬') || hasProgressMarker(content)) {
    return { phase: 'running', taskId, progress: progress > 0 ? progress : 5, stepLabel, message: content }
  }

  return null
}

const PHASE_ICONS: Record<string, JSX.Element> = {
  extracting_text: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
      <polyline points="13 2 13 9 20 9"/>
    </svg>
  ),
  searching_patents: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8"/>
      <path d="m21 21-4.3-4.3"/>
    </svg>
  ),
  generating_columns: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="7" height="18" rx="1"/>
      <rect x="14" y="3" width="7" height="7" rx="1"/>
      <rect x="14" y="14" width="7" height="7" rx="1"/>
    </svg>
  ),
  analyzing: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21.21 15.89A10 10 0 1 1 8 2.83"/>
      <path d="M22 12A10 10 0 0 0 12 2v10z"/>
    </svg>
  ),
  generating_report: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
      <polyline points="14 2 14 8 20 8"/>
      <line x1="8" y1="13" x2="16" y2="13"/>
      <line x1="8" y1="17" x2="12" y2="17"/>
    </svg>
  ),
  exporting: (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
  ),
}

const PHASES = [
  { key: 'extracting_text', label: '文件解析' },
  { key: 'searching_patents', label: '检索' },
  { key: 'generating_columns', label: '分析框架' },
  { key: 'analyzing', label: '专利分析' },
  { key: 'generating_report', label: '报告撰写' },
  { key: 'exporting', label: '文件导出' },
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
            if (p.key === 'extracting_text' && state.progress >= 20) status = 'done'
            else if (p.key === 'extracting_text' && state.progress >= 0) status = 'active'
            else if (p.key === 'searching_patents' && state.progress >= 2) status = 'done'
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
                <span className="lt-phase-icon">{PHASE_ICONS[p.key] || null}</span>
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

      {/* Completed state — no download buttons; reports are accessed elsewhere */}
      {state.phase === 'completed' && (
        <p className="lt-current-step">✅ 分析完成</p>
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
