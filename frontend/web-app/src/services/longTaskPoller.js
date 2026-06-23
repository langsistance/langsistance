import { useState, useEffect, useRef, useCallback } from 'react'
import { pollTaskStatus } from './sessionService'

/**
 * Polling states for a long task.
 *
 * @typedef {'idle'|'polling'|'completed'|'error'} PollState
 */

/**
 * Default phase labels used when task phases are not available.
 */
const DEFAULT_PHASES = [
  { key: 'pending', label: '任务排队中' },
  { key: 'running', label: '分析进行中' },
  { key: 'generating', label: '报告生成中' },
  { key: 'completed', label: '任务完成' },
]

/**
 * Convert backend status string to a normalized phase index.
 */
function statusToPhaseIndex(status) {
  switch (status) {
    case 'pending': return 0
    case 'running':
    case 'analyzing':
    case 'searching': return 1
    case 'generating':
    case 'writing':
    case 'formatting': return 2
    case 'completed':
    case 'success': return 3
    default: return -1
  }
}

/**
 * Hook that polls GET /long_task/{task_id}/status until the task completes or errors.
 *
 * @param {string|null} taskId   — the long task ID to poll
 * @param {object}      [opts]
 * @param {number}      [opts.interval=2000]       — base poll interval in ms
 * @param {number}      [opts.maxInterval=30000]   — max poll interval after backoff
 * @param {number}      [opts.backoffFactor=1.5]   — exponential backoff multiplier
 * @returns {{ status: PollState, progress: object|null, error: string|null, pollResult: object|null, startPolling: (id: string) => void, stopPolling: () => void }}
 */
export function useLongTaskPoller(taskId, opts = {}) {
  const {
    interval: baseInterval = 2000,
    maxInterval = 30000,
    backoffFactor = 1.5,
  } = opts

  const [pollState, setPollState] = useState('idle')
  const [progress, setProgress] = useState(null)
  const [error, setError] = useState(null)
  const [pollResult, setPollResult] = useState(null)
  const currentTaskId = useRef(taskId)
  const pollTimer = useRef(null)
  const backoffRef = useRef(baseInterval)

  const stopPolling = useCallback(() => {
    if (pollTimer.current) {
      clearTimeout(pollTimer.current)
      pollTimer.current = null
    }
    backoffRef.current = baseInterval
  }, [baseInterval])

  const startPolling = useCallback((id) => {
    currentTaskId.current = id
    setPollState('polling')
    setError(null)
    setPollResult(null)
    setProgress(null)
    backoffRef.current = baseInterval
  }, [baseInterval])

  // Core poll loop
  useEffect(() => {
    if (pollState !== 'polling' || !currentTaskId.current) return

    let cancelled = false

    async function poll() {
      try {
        const res = await pollTaskStatus(currentTaskId.current)
        if (cancelled) return

        const data = res?.data || res

        // Build a normalized progress snapshot
        const snapshot = {
          status: data.status || 'pending',
          progress: data.progress != null ? data.progress : null,
          progressLabel: data.progress_label || '',
          message: data.message || data.status_message || '',
          phases: data.phases || DEFAULT_PHASES,
          currentPhase: data.current_phase != null ? data.current_phase : statusToPhaseIndex(data.status),
          steps: data.steps || [],
          results: data.results || data.partial_results || [],
          error: data.error || null,
        }

        setProgress(snapshot)

        if (data.status === 'completed' || data.status === 'success') {
          setPollState('completed')
          setPollResult(data)
          return
        }

        if (data.status === 'failed' || data.status === 'error') {
          setPollState('error')
          setError(data.error || data.message || 'Task failed')
          return
        }

        // Schedule next poll with backoff
        const nextInterval = Math.min(backoffRef.current, maxInterval)
        pollTimer.current = setTimeout(() => {
          backoffRef.current = Math.min(backoffRef.current * backoffFactor, maxInterval)
          poll()
        }, nextInterval)
      } catch (err) {
        if (cancelled) return
        setError(err.message || 'Poll request failed')
        setPollState('error')
      }
    }

    // Start first poll immediately
    backoffRef.current = baseInterval
    pollTimer.current = setTimeout(poll, 200)

    return () => {
      cancelled = true
      stopPolling()
    }
  }, [pollState, baseInterval, maxInterval, backoffFactor, stopPolling])

  // Cleanup on unmount
  useEffect(() => {
    return () => stopPolling()
  }, [stopPolling])

  return { status: pollState, progress, error, pollResult, startPolling, stopPolling }
}
