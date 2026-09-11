import { request } from './api'

/** 与 sources/long_task/status_manager.py 的状态值一致。 */
export type TaskStatus =
  | 'pending'
  | 'queued'
  | 'running'
  | 'paused'
  | 'cancelling'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'unknown'

export interface TaskReportFile {
  format: string
  filename: string
  size: number
}

export interface TaskState {
  taskId: string
  status: TaskStatus
  phase: string
  progress: number
  step: string
  reportFiles: TaskReportFile[]
  error: string
}

/** web 同节奏（frontend/nextjs/lib/useChatStream.ts:629-630） */
export const POLL_INTERVAL_MS = 1500
/** 连续失败这么多次后判定"状态未知"并停止轮询 */
export const MAX_POLL_FAILURES = 5

/** 阶段文案对齐 web 的 STANDARD_PHASES（LongTaskProgress.tsx:224-231） */
export const PHASE_LABELS: Record<string, string> = {
  extracting_text: '提取文本',
  searching_patents: '检索专利',
  generating_columns: '生成分析维度',
  analyzing: '逐件分析',
  generating_report: '生成报告',
  exporting: '导出文件',
}

const TERMINAL: TaskStatus[] = ['completed', 'failed', 'cancelled', 'unknown']

export function isTerminal(status: TaskStatus): boolean {
  return TERMINAL.indexOf(status) >= 0
}

export function phaseLabel(phase: string): string {
  return PHASE_LABELS[phase] || phase || '处理中'
}

/** 把后端 status 载荷归一成 TaskState。 */
export function toTaskState(taskId: string, raw: Record<string, any>): TaskState {
  const files = Array.isArray(raw.report_files) ? raw.report_files : []
  return {
    taskId,
    status: (raw.status || 'unknown') as TaskStatus,
    phase: String(raw.current_phase || ''),
    progress: Number(raw.progress || 0),
    step: String(raw.current_step || ''),
    reportFiles: files.map((f: any) => ({
      format: String(f.format || ''),
      filename: String(f.filename || ''),
      size: Number(f.size || 0),
    })),
    error: String(raw.error_message || ''),
  }
}

/** 批量取状态。后端只返回属于本人的任务（api_routes/long_task.py:561+），
 *  故返回的键可能少于传入的 taskIds。 */
export async function pollTasks(
  taskIds: string[],
): Promise<Record<string, TaskState>> {
  if (taskIds.length === 0) return {}
  const body = await request<{ statuses: Record<string, any> }>(
    '/long_task/batch_status',
    { method: 'POST', data: { task_ids: taskIds.slice(0, 20) } },
  )
  const out: Record<string, TaskState> = {}
  const statuses = (body && body.statuses) || {}
  for (const tid of taskIds.slice(0, 20)) {
    const raw = statuses[tid]
    if (raw) out[tid] = toTaskState(tid, raw)
  }
  return out
}

/** 一键重试，返回新任务的 task_id（api_routes/long_task.py:426-541）。 */
export async function retryTask(taskId: string): Promise<string> {
  const body = await request<{ task_id: string }>(
    `/long_task/${taskId}/retry`,
    { method: 'POST' },
  )
  return body.task_id
}
