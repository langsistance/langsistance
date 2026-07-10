import { recoverLongTaskByQueryId } from '@/services/api'

export interface RecoveredLongTask {
  taskId: string
  sessionId: string
  status: string
}

/** Poll backend until a long task registered for query_id appears (SSE may have timed out). */
export async function pollRecoverLongTask(
  queryId: string,
  maxAttempts = 15,
  intervalMs = 2000,
): Promise<RecoveredLongTask | null> {
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      const data = await recoverLongTaskByQueryId(queryId)
      if (data.found && data.task_id) {
        return {
          taskId: data.task_id,
          sessionId: data.session_id || '',
          status: data.status || 'running',
        }
      }
    } catch {
      // Retry — backend may still be creating the task
    }
    if (attempt < maxAttempts - 1) {
      await new Promise((resolve) => setTimeout(resolve, intervalMs))
    }
  }
  return null
}
