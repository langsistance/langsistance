import { useState, useEffect, useCallback } from 'react'

/* ------------------------------------------------------------------ */
/*  Status icons                                                      */
/* ------------------------------------------------------------------ */

function IconCompleted() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10A37F" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="16 8 10 16 7 13" />
    </svg>
  )
}

function IconRunning() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563EB" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  )
}

function IconPending() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9CA3AF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <circle cx="12" cy="12" r="3" fill="#9CA3AF" />
    </svg>
  )
}

function IconError() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#DC2626" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="15" y1="9" x2="9" y2="15" />
      <line x1="9" y1="9" x2="15" y2="15" />
    </svg>
  )
}

/* ------------------------------------------------------------------ */
/*  Progress bar                                                      */
/* ------------------------------------------------------------------ */

function ProgressBar({ progress }) {
  const pct = progress != null ? Math.min(Math.max(progress, 0), 100) : 0
  return (
    <div style={{
      width: '100%',
      height: 6,
      background: '#E5E7EB',
      borderRadius: 3,
      overflow: 'hidden',
      marginTop: 8,
    }}>
      <div style={{
        width: `${pct}%`,
        height: '100%',
        background: 'linear-gradient(90deg, #2563EB, #10A37F)',
        borderRadius: 3,
        transition: 'width 0.5s ease',
      }} />
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Phase list                                                        */
/* ------------------------------------------------------------------ */

function PhaseRow({ phase, index, currentPhaseIndex }) {
  let icon
  let labelClass = {}

  if (index < currentPhaseIndex) {
    icon = <IconCompleted />
    labelClass = { color: '#10A37F', fontWeight: 500 }
  } else if (index === currentPhaseIndex) {
    icon = <IconRunning />
    labelClass = { color: '#2563EB', fontWeight: 600 }
  } else {
    icon = <IconPending />
    labelClass = { color: '#9CA3AF' }
  }

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 10,
      padding: '6px 0',
    }}>
      <div style={{ flexShrink: 0, display: 'flex', alignItems: 'center' }}>
        {icon}
        {index < (phase?.length ?? 1) - 1 && (
          <div style={{
            width: 1,
            height: 16,
            background: index < currentPhaseIndex ? '#10A37F' : '#E5E7EB',
            marginLeft: 9,
          }} />
        )}
      </div>
      <span style={{ fontSize: 14, ...labelClass }}>
        {phase.label || phase.key || `Phase ${index + 1}`}
      </span>
    </div>
  )
}

function PhaseList({ phases, currentPhaseIndex }) {
  if (!phases || phases.length === 0) return null
  return (
    <div style={{ marginTop: 12 }}>
      {phases.map((phase, i) => (
        <PhaseRow
          key={phase.key || i}
          phase={phase}
          index={i}
          currentPhaseIndex={currentPhaseIndex}
        />
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Steps list                                                        */
/* ------------------------------------------------------------------ */

function StepIcon({ status }) {
  if (status === 'completed') return <IconCompleted />
  if (status === 'running' || status === 'in_progress') return <IconRunning />
  if (status === 'failed' || status === 'error') return <IconError />
  return <IconPending />
}

function StepsList({ steps }) {
  if (!steps || steps.length === 0) return null
  return (
    <div style={{
      marginTop: 8,
      paddingLeft: 28,
      borderLeft: '2px solid #E5E7EB',
      marginLeft: 8,
    }}>
      <p style={{ fontSize: 12, color: '#6B7280', marginBottom: 6, fontWeight: 500 }}>详细步骤</p>
      {steps.map((step, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '3px 0' }}>
          <StepIcon status={step.status || step.state} />
          <span style={{ fontSize: 13, color: '#374151' }}>
            {step.label || step.description || `Step ${i + 1}`}
          </span>
          {step.duration && (
            <span style={{ fontSize: 11, color: '#9CA3AF', marginLeft: 'auto' }}>
              {step.duration}
            </span>
          )}
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Dynamic results table                                             */
/* ------------------------------------------------------------------ */

function ResultsTable({ results, columns }) {
  if (!results || results.length === 0) return null

  // If columns are not specified, derive from first result's keys
  const derivedCols = columns || (results[0] ? Object.keys(results[0]).filter(
    (k) => typeof results[0][k] !== 'object' || results[0][k] === null
  ) : [])
  if (derivedCols.length === 0) return null

  return (
    <div style={{ overflowX: 'auto', marginTop: 12, borderRadius: 8, border: '1px solid #E5E7EB' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ background: '#F9FAFB' }}>
            {derivedCols.map((col) => (
              <th key={col} style={{ padding: '8px 12px', textAlign: 'left', color: '#6B7280', fontWeight: 500, borderBottom: '1px solid #E5E7EB' }}>
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {results.map((row, ri) => (
            <tr key={ri} style={{ borderBottom: ri < results.length - 1 ? '1px solid #F3F4F6' : 'none' }}>
              {derivedCols.map((col) => (
                <td key={col} style={{ padding: '8px 12px', color: '#374151' }}>
                  {row[col] != null ? String(row[col]) : '-'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Main component                                                    */
/* ------------------------------------------------------------------ */

/**
 * LongTaskProgress — displays real-time progress of a long-running task.
 *
 * Props:
 *   progress       — object returned by useLongTaskPoller (progress snapshot)
 *   status         — poller status string ('idle'|'polling'|'completed'|'error')
 *   error          — error message string (when status='error')
 *   pollResult     — final result data (when status='completed')
 *   columns        — optional column list for the results table (auto-detected if omitted)
 *   onDownload     — callback(format) triggered when user clicks download
 *   onDismiss      — callback() to close/dismiss the panel
 */
export default function LongTaskProgress({
  progress,
  status,
  error: pollError,
  pollResult,
  columns,
  onDownload,
  onDismiss,
}) {
  const [tableCols, setTableCols] = useState(columns)

  // Auto-detect columns from results when they arrive
  useEffect(() => {
    if (!columns && progress?.results && progress.results.length > 0) {
      const keys = Object.keys(progress.results[0]).filter(
        (k) => typeof progress.results[0][k] !== 'object' || progress.results[0][k] === null
      )
      setTableCols(keys)
    }
  }, [progress?.results, columns])

  if (status === 'idle') return null

  return (
    <div style={{
      background: '#FFFFFF',
      border: '1px solid #E5E7EB',
      borderRadius: 12,
      padding: '16px 20px',
      margin: '12px 0',
      boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {status === 'polling' && <div className="loading" style={{ width: 14, height: 14, borderWidth: 2 }} />}
          {status === 'completed' && <IconCompleted />}
          {status === 'error' && <IconError />}
          <span style={{ fontWeight: 600, fontSize: 15, color: '#111827' }}>
            {status === 'polling' && (progress?.message || '任务进行中...')}
            {status === 'completed' && '任务已完成'}
            {status === 'error' && '任务失败'}
          </span>
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: '#9CA3AF',
              padding: 4,
              display: 'flex',
              alignItems: 'center',
              borderRadius: 4,
            }}
            title="关闭"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        )}
      </div>

      {/* Progress bar */}
      {progress?.progress != null && status === 'polling' && (
        <ProgressBar progress={progress.progress} />
      )}

      {/* Error */}
      {status === 'error' && (
        <p style={{ color: '#DC2626', fontSize: 13, marginTop: 8 }}>
          {pollError || progress?.error || '未知错误'}
        </p>
      )}

      {/* Phase list */}
      <PhaseList
        phases={progress?.phases}
        currentPhaseIndex={progress?.currentPhase ?? -1}
      />

      {/* Steps */}
      <StepsList steps={progress?.steps} />

      {/* Results table (incremental) */}
      {progress?.results && progress.results.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <p style={{ fontSize: 12, color: '#6B7280', fontWeight: 500, marginBottom: 4 }}>
            分析结果 ({progress.results.length})
          </p>
          <ResultsTable results={progress.results} columns={tableCols} />
        </div>
      )}

      {/* Download buttons (on completion) */}
      {status === 'completed' && onDownload && (
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <button
            className="btn btn-primary"
            onClick={() => onDownload('pdf')}
            style={{ fontSize: 13, padding: '6px 14px' }}
          >
            下载 PDF 报告
          </button>
          <button
            className="btn btn-secondary"
            onClick={() => onDownload('docx')}
            style={{ fontSize: 13, padding: '6px 14px' }}
          >
            下载 DOCX 报告
          </button>
        </div>
      )}
    </div>
  )
}
