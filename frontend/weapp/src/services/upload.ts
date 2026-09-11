import Taro from '@tarojs/taro'
import { API_BASE, STORAGE_KEYS } from '../config'
import { clearAuthAndRedirect } from './api'
import { ChatMsg } from './chat'
import { SseParser } from './chatStream'
import { ensurePrivacy } from './privacy'

/** 与 web 及后端同值：api_routes/core.py:953-954、nextjs useChatStream.ts:64 */
export const MAX_FILE_BYTES = 10 * 1024 * 1024
/** 后端丢弃 <50 字节的文件（api_routes/core.py:967），前端提前拦 */
export const MIN_FILE_BYTES = 50
export const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.xml']

export interface PickedFile {
  path: string
  name: string
  size: number
}

export interface UploadOutcome {
  taskId: string
  sessionId: string
  patentCount: number
}

export function extensionOf(name: string): string {
  const i = name.lastIndexOf('.')
  return i < 0 ? '' : name.slice(i).toLowerCase()
}

/** 通过返回 null；不通过返回给用户看的原因。 */
export function validateFile(file: { name: string; size: number }): string | null {
  if (!ALLOWED_EXTENSIONS.includes(extensionOf(file.name))) {
    return '仅支持 PDF / DOCX / XML 格式'
  }
  if (file.size > MAX_FILE_BYTES) return '文件超过 10MB 上限'
  if (file.size < MIN_FILE_BYTES) return '文件内容为空或过小'
  return null
}

/**
 * 选一个文件。内部先过隐私授权——wx.chooseMessageFile 是隐私接口，
 * 未授权会被微信直接禁用，先选后授权会让用户白选一次。
 * 用户取消或校验不过返回 null（已 toast 提示）。
 */
export async function pickFile(): Promise<PickedFile | null> {
  // ensurePrivacy 返回 false = 用户拒绝隐私授权。必须中止：chooseMessageFile
  // 是隐私接口，未获授权时调用无意义。（Task 4 的评审把它的返回类型从
  // Promise<void> 改成了 Promise<boolean>——忽略了返回值就等于没做这道闸。）
  const allowed = await ensurePrivacy()
  if (!allowed) {
    Taro.showToast({ title: '需同意隐私保护指引后才能上传文件', icon: 'none' })
    return null
  }
  let res: Taro.chooseMessageFile.SuccessCallbackResult
  try {
    res = await Taro.chooseMessageFile({
      count: 1,
      type: 'file',
      extension: ['pdf', 'docx', 'xml'],
    })
  } catch {
    return null // 用户取消
  }
  const f = res.tempFiles && res.tempFiles[0]
  if (!f) return null
  const picked: PickedFile = { path: f.path, name: f.name, size: f.size }
  const reason = validateFile(picked)
  if (reason) {
    Taro.showToast({ title: reason, icon: 'none' })
    return null
  }
  return picked
}

/**
 * 上传单个文件并发起长任务。字段名与 web 完全一致
 * （frontend/nextjs/services/api.ts:139-146）。
 *
 * 注意 conversation_history 必须**包含本轮新提问**——后端会把它直接写成
 * 会话的 messages（api_routes/core.py:1041），不含新提问则该轮不落库。
 */
export async function uploadQuery(
  file: PickedFile,
  query: string,
  conversationHistory: ChatMsg[],
  queryId: string,
  sessionId?: string,
  onProgress?: (percent: number) => void,
): Promise<UploadOutcome> {
  const token = Taro.getStorageSync(STORAGE_KEYS.wxToken)
  const formData: Record<string, string> = {
    query,
    query_id: queryId,
    // multipart 的 formData 值只能是字符串，故此处是 '2' 而非数字 2，
    // 与 web 的 formData 分支一致（nextjs services/api.ts:141）
    push_filter: '2',
    conversation_history: JSON.stringify(conversationHistory),
  }
  if (sessionId) formData.session_id = sessionId

  const task = Taro.uploadFile({
    url: `${API_BASE}/query_stream`,
    filePath: file.path,
    name: 'patent_files',
    formData,
    header: { Authorization: `Bearer ${token}` },
    timeout: 120000,
  })
  if (onProgress && typeof task.onProgressUpdate === 'function') {
    task.onProgressUpdate((r) => onProgress(r.progress))
  }

  let res: Taro.uploadFile.SuccessCallbackResult
  try {
    res = await task
  } catch (err: any) {
    // Taro 的 promise 包装在失败时 reject 的是微信原始 fail 对象（不是 Error），
    // 其上**没有 statusCode**——若直接往下走，401 与 >=400 两个判断都是 false，
    // 会一路落到"未返回任务号"，把网络失败报成假的成功，且 401 分支永远不可达。
    throw new Error((err && err.errMsg) || '网络错误，请检查连接后重试')
  }
  const raw = String(res.data || '')

  if (res.statusCode === 401) clearAuthAndRedirect()
  if (res.statusCode >= 400) {
    // 后端错误体是 JSON（FastAPI HTTPException），尽力取可读文案
    let detail = ''
    try {
      const parsed = JSON.parse(raw)
      // 后端 HTTPException 用 detail；文件上传分支的 500 用 error（core.py:1233-1242）
      detail = parsed.detail || parsed.message || parsed.error || ''
    } catch {
      detail = ''
    }
    throw new Error(detail || `上传失败(${res.statusCode})`)
  }

  for (const ev of new SseParser().push(raw)) {
    if (ev.type === 'long_task_created') {
      const taskId = String(ev.task_id || '')
      if (!taskId) {
        // 帧畸形：返回空 taskId 会让调用方去轮询一个不存在的任务号
        throw new Error('后端未返回任务号，请稍后在历史会话中查看')
      }
      return {
        taskId,
        sessionId: String(ev.session_id || ''),
        patentCount: Number(ev.patent_count || 0),
      }
    }
    if (ev.type === 'error') {
      throw new Error(String(ev.message || ev.content || '文件处理失败'))
    }
  }
  throw new Error('上传已完成但未返回任务号，请稍后在历史会话中查看')
}
