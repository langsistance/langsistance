'use client'

import { useEffect, useRef, useState } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { submitLongTask, pollLongTaskBatchStatus, getLongTaskReportUrl } from '@/services/api'
import { useChatSession } from '@/contexts/ChatContext'
import LongTaskProgress from '@/components/app/LongTaskProgress'

interface TaskState {
  taskId: string
  kind: 'prosecution' | 'family'
  status: string
  progress: number | null
  currentStep: string
  resultSummary: string | null
  error: string | null
}

export default function ProsecutionTab({ row }: { row: any }) {
  const { t } = useI18n()
  const { sessionId } = useChatSession()
  const [tasks, setTasks] = useState<TaskState[]>([])
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState(false)
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const hasUsAppNumber = /^\d{8}$/.test(row.applicationNumber || '')
  const hasAnyId = Boolean(row.patentId || row.applicationNumber)

  async function handleSubmit(kind: 'prosecution' | 'family') {
    setSubmitting(true)
    setSubmitError(false)
    try {
      const res = await submitLongTask({
        scenario: kind,
        patentId: kind === 'prosecution' ? row.applicationNumber : (row.patentId || row.applicationNumber),
        query: kind === 'prosecution'
          ? `分析专利 ${row.patentId || row.applicationNumber} 的审查历史`
          : `分析 ${row.patentId || row.applicationNumber} 及其全球同族申请的审查差异`,
        lang: 'zh',
        ...(sessionId ? { sessionId } : {}),
      })
      setTasks((prev) => [...prev, {
        taskId: res.task_id, kind, status: res.status, progress: 0,
        currentStep: '', resultSummary: null, error: null,
      }])
      setActiveTaskId(res.task_id)
    } catch {
      setSubmitError(true)
    } finally {
      setSubmitting(false)
    }
  }

  useEffect(() => {
    const openTasks = tasks.filter((task) => !['completed', 'failed'].includes(task.status))
    if (openTasks.length === 0) return

    async function pollAll() {
      const ids = openTasks.map((task) => task.taskId)
      try {
        const batch = await pollLongTaskBatchStatus(ids)
        setTasks((prev) => prev.map((task) => {
          const data = batch[task.taskId]
          if (!data) return task
          if (data.status === 'completed' || data.status === 'success') {
            return { ...task, status: 'completed', progress: 100, resultSummary: data.result_summary || task.resultSummary }
          }
          if (data.status === 'failed' || data.status === 'error') {
            return { ...task, status: 'failed', error: data.error_message || 'failed' }
          }
          return { ...task, status: data.status, progress: data.progress ?? task.progress, currentStep: data.current_step || task.currentStep }
        }))
      } catch {
        // Transient poll error — keep polling
      }
    }

    pollAll()
    pollTimerRef.current = setInterval(pollAll, 1500)
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [tasks.map((task) => task.taskId + task.status).join(',')])

  const activeTask = tasks.find((task) => task.taskId === activeTaskId) || tasks[tasks.length - 1]

  return (
    <div className="results-detail-card">
      <div className="results-prosecution-entries">
        <button
          className="results-prosecution-entry"
          disabled={!hasUsAppNumber || submitting}
          title={hasUsAppNumber ? '' : t('results.prosecutionUsUnavailable')}
          onClick={() => handleSubmit('prosecution')}
        >
          🇺🇸 {t('results.prosecutionUs')}
        </button>
        <button
          className="results-prosecution-entry"
          disabled={!hasAnyId || submitting}
          title={hasAnyId ? '' : t('results.prosecutionNoId')}
          onClick={() => handleSubmit('family')}
        >
          🌐 {t('results.prosecutionFamily')}
        </button>
      </div>

      {submitting && <p>{t('results.prosecutionSubmitting')}</p>}
      {submitError && <p className="results-error-text">{t('results.prosecutionSubmitError')}</p>}

      {tasks.length > 1 && (
        <div className="results-prosecution-switch">
          {tasks.map((task) => (
            <button
              key={task.taskId}
              className={activeTask?.taskId === task.taskId ? 'active' : ''}
              onClick={() => setActiveTaskId(task.taskId)}
            >
              {task.kind === 'prosecution' ? '🇺🇸' : '🌐'} {task.status}
            </button>
          ))}
        </div>
      )}

      {activeTask && (
        <div className="results-prosecution-task">
          {activeTask.status === 'failed' ? (
            <p className="results-error-text">{activeTask.error || t('results.prosecutionSubmitError')}</p>
          ) : activeTask.status === 'completed' ? (
            <div>
              {activeTask.resultSummary && (
                <LongTaskProgress
                  content={activeTask.resultSummary}
                  resultSummary={activeTask.resultSummary}
                  streaming={false}
                />
              )}
              <div className="results-prosecution-downloads">
                <a href={getLongTaskReportUrl(activeTask.taskId, 'pdf')} download>PDF</a>
                <a href={getLongTaskReportUrl(activeTask.taskId, 'docx')} download>Word</a>
              </div>
            </div>
          ) : (
            <LongTaskProgress
              content={`${t('results.prosecutionRunning')} [${activeTask.progress ?? 0}%] ${activeTask.currentStep} Task ID: ${activeTask.taskId}`}
              streaming={true}
            />
          )}
        </div>
      )}
    </div>
  )
}
