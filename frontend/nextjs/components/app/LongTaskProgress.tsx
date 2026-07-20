'use client'

import { useMemo, useState } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { renderMarkdownToHtml } from '@/lib/markdownRender'

interface Props {
  content: string
  resultSummary?: string
  streaming: boolean
}

interface TaskState {
  phase: 'submitted' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'
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

  const isLongTask =
    content.includes('🔬') ||
    content.includes('✅') ||
    content.includes('❌') ||
    content.includes('⏸') ||
    content.includes('⏹') ||
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
  // Clean garbled Unicode from LLM output
  stepLabel = stepLabel.replace(/\uFFFD/g, '')
    .replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])/g, '')
    .replace(/(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, '')
    .replace(/[\u0080-\u009F]/g, '')

  // Paused state
  if (content.includes('⏸') || content.includes('已暂停')) {
    return {
      phase: 'paused',
      taskId,
      progress,
      stepLabel,
      message: content,
    }
  }

  // Cancelled state
  if (content.includes('⏹') || content.includes('已取消') || content.includes('已停止')) {
    return {
      phase: 'cancelled',
      taskId,
      progress: 0,
      stepLabel: '',
      message: content,
    }
  }

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

  // Running: has 🔬 marker or progress percentage
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

const PHASE_LABEL_KEYS: Record<string, string> = {
  extracting_text: 'longTask.phaseExtractingText',
  searching_patents: 'longTask.phaseSearchingPatents',
  generating_columns: 'longTask.phaseGeneratingColumns',
  analyzing: 'longTask.phaseAnalyzing',
  generating_report: 'longTask.phaseGeneratingReport',
  exporting: 'longTask.phaseExporting',
}

const PHASES = [
  { key: 'extracting_text', fileUploadOnly: true },
  { key: 'searching_patents' },
  { key: 'generating_columns' },
  { key: 'analyzing' },
  { key: 'generating_report' },
  { key: 'exporting' },
]

const PHASE_MATCH_KEYWORDS: Record<string, string[]> = {
  extracting_text: ['文件解析', '解析上传', 'Parsing', 'Extracting', 'OCR'],
  searching_patents: ['检索', 'Searching', 'Fetching USPTO'],
  generating_columns: ['分析框架', '分析维度', 'framework', 'Analysis framework', 'Building analysis'],
  analyzing: ['正在分析', '下载专利', '专利分析', '已完成', 'Analyzing', 'Downloading', 'Analysis progress', 'Completed'],
  generating_report: ['报告', '撰写', 'Report', 'Writing', 'summary', 'outline'],
  exporting: ['Word', 'PDF', '导出', 'Generating Word', 'Converting DOCX', 'Exporting'],
}

function isFileUploadMode(content: string): boolean {
  return content.includes('上传文件') || content.includes('extracting_text') || content.includes('uploaded file') || content.includes('Parsing')
}

async function callLongTaskApi(taskId: string, action: 'pause' | 'resume' | 'stop'): Promise<boolean> {
  try {
    const { getValidToken } = await import('@/lib/auth-client')
    const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'https://api.copiioai.com'
    const token = await getValidToken()
    if (!token) return false
    const res = await fetch(`${API_BASE}/long_task/${taskId}/${action}`, {
      method: 'POST',
      headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    })
    const data = await res.json()
    return data.success === true
  } catch {
    return false
  }
}

export default function LongTaskProgress({ content, resultSummary, streaming }: Props) {
  const { t } = useI18n()
  const state = parseTaskContent(content)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const summaryHtml = useMemo(
    () => (resultSummary ? renderMarkdownToHtml(resultSummary) : ''),
    [resultSummary],
  )
  const summaryStreaming = Boolean(
    resultSummary
    && (state?.phase === 'running' || state?.phase === 'paused')
    && state.progress >= 76
    && state.progress < 100,
  )

  if (!state) return null

  async function handleAction(action: 'pause' | 'resume' | 'stop') {
    if (!state?.taskId) return
    setActionLoading(action)
    await callLongTaskApi(state.taskId, action)
    setActionLoading(null)
  }

  return (
    <div className="lt-progress-card">
      {/* Header */}
      <div className="lt-progress-header">
        <div className="lt-progress-pulse" data-active={
          state.phase === 'running' || state.phase === 'submitted'
        } />
        <span className="lt-progress-title">
          {state.phase === 'completed'
            ? t('longTask.titleCompleted')
            : state.phase === 'failed'
            ? t('longTask.titleFailed')
            : state.phase === 'cancelled'
            ? t('longTask.titleCancelled')
            : state.phase === 'paused'
            ? t('longTask.titlePaused')
            : state.phase === 'submitted'
            ? t('longTask.titleSubmitted')
            : t('longTask.titleRunning')}
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

      {/* Paused progress bar (frozen at pause point) */}
      {state.phase === 'paused' && (
        <div className="lt-progress-bar-wrap" style={{ opacity: 0.6 }}>
          <div className="lt-progress-bar-track">
            <div
              className="lt-progress-bar-fill paused"
              style={{ width: `${Math.max(state.progress, 2)}%` }}
            />
          </div>
          <span className="lt-progress-pct">{state.progress}%</span>
        </div>
      )}

      {/* Phase indicators + inline action buttons */}
      {(state.phase === 'running' || state.phase === 'submitted' || state.phase === 'paused') && (
        <div className="lt-phases">
          {PHASES.filter(p => !p.fileUploadOnly || isFileUploadMode(content)).map((p) => {
            let status: 'done' | 'active' | 'pending' = 'pending'
            if (p.key === 'extracting_text' && state.progress >= 20) status = 'done'
            else if (p.key === 'extracting_text' && state.progress >= 0) status = 'active'
            else if (p.key === 'searching_patents' && state.progress >= 2) status = 'done'
            else if (p.key === 'generating_columns' && state.progress >= 5) status = 'done'
            else if (p.key === 'analyzing' && state.progress >= 10) status = state.progress < 75 ? 'active' : 'done'
            else if (p.key === 'generating_report' && state.progress >= 80) status = state.progress < 90 ? 'active' : 'done'
            else if (p.key === 'exporting' && state.progress >= 92) status = 'active'

            const phaseLabel = t(PHASE_LABEL_KEYS[p.key])
            const keywords = PHASE_MATCH_KEYWORDS[p.key] || [phaseLabel]
            if (keywords.some(kw => state.stepLabel.includes(kw)) && status !== 'done') status = 'active'

            return (
              <div
                key={p.key}
                className={`lt-phase-dot ${status}`}
                title={phaseLabel}
              >
                <span className="lt-phase-icon">{PHASE_ICONS[p.key] || null}</span>
                <span className="lt-phase-label">{phaseLabel}</span>
              </div>
            )
          })}

          {/* Inline icon-only action buttons — like video player controls */}
          {state.taskId && (
            <>
              {/* Running: pause + stop */}
              {state.phase === 'running' && (
                <>
                  <span className="lt-phase-sep" />
                  <button
                    className="lt-phase-action lt-phase-action-pause"
                    onClick={() => handleAction('pause')}
                    disabled={actionLoading !== null || streaming}
                    title={t('longTask.actionPause')}
                  >
                    {actionLoading === 'pause' ? (
                      <span className="lt-btn-spinner" />
                    ) : (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                        <rect x="6" y="4" width="4" height="16" rx="1"/>
                        <rect x="14" y="4" width="4" height="16" rx="1"/>
                      </svg>
                    )}
                  </button>
                  <button
                    className="lt-phase-action lt-phase-action-stop"
                    onClick={() => handleAction('stop')}
                    disabled={actionLoading !== null || streaming}
                    title={t('longTask.actionStop')}
                  >
                    {actionLoading === 'stop' ? (
                      <span className="lt-btn-spinner" />
                    ) : (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                        <rect x="5" y="5" width="14" height="14" rx="1"/>
                      </svg>
                    )}
                  </button>
                </>
              )}

              {/* Paused: play + stop */}
              {state.phase === 'paused' && (
                <>
                  <span className="lt-phase-sep" />
                  <button
                    className="lt-phase-action lt-phase-action-resume"
                    onClick={() => handleAction('resume')}
                    disabled={actionLoading !== null}
                    title={t('longTask.actionResume')}
                  >
                    {actionLoading === 'resume' ? (
                      <span className="lt-btn-spinner" />
                    ) : (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                        <polygon points="7,4 20,12 7,20"/>
                      </svg>
                    )}
                  </button>
                  <button
                    className="lt-phase-action lt-phase-action-stop"
                    onClick={() => handleAction('stop')}
                    disabled={actionLoading !== null}
                    title={t('longTask.actionStop')}
                  >
                    {actionLoading === 'stop' ? (
                      <span className="lt-btn-spinner" />
                    ) : (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                        <rect x="5" y="5" width="14" height="14" rx="1"/>
                      </svg>
                    )}
                  </button>
                </>
              )}
            </>
          )}
        </div>
      )}

      {/* Current step */}
      {state.stepLabel && (state.phase === 'running' || state.phase === 'paused') && (
        <p className="lt-current-step">{state.stepLabel}</p>
      )}

      {/* Report summary preview (streamed during Phase 3, shown before downloads) */}
      {resultSummary && (
        <div className={`lt-summary${summaryStreaming ? ' lt-summary-streaming' : ''}`}>
          <div className="lt-summary-header">
            <span className="lt-summary-title">{t('longTask.summaryTitle')}</span>
            {summaryStreaming && (
              <span className="lt-summary-badge">{t('longTask.summaryStreaming')}</span>
            )}
          </div>
          <div
            className="lt-summary-body chat-markdown"
            dangerouslySetInnerHTML={{ __html: summaryHtml }}
          />
          {summaryStreaming && <span className="lt-summary-cursor" aria-hidden="true">▋</span>}
        </div>
      )}

      {/* Cancelled: info label */}
      {state.phase === 'cancelled' && state.taskId && (
        <p className="lt-cancelled-label">{t('longTask.cancelledLabel')}</p>
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
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                  <path d="M12 18v-6"/>
                  <path d="m9 15 3 3 3-3"/>
                </svg>
              </span>
              <span className="lt-dl-label">
                {t('longTask.downloadLabel', { format: link.label })}
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
        <p className="lt-current-step">{t('longTask.submittedMessage')}</p>
      )}
    </div>
  )
}
