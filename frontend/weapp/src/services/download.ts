import Taro from '@tarojs/taro'
import { API_BASE, STORAGE_KEYS } from '../config'
import { clearAuthAndRedirect } from './api'

/**
 * wx.openDocument 的 fileType 白名单。
 * 官方合法值只有 doc/docx/xls/xlsx/ppt/pptx/pdf——**没有 csv**，
 * 也没有 md。这些格式只能走 shareFileMessage 转发到聊天。
 *
 * 同时作为 openDocument 的 fileType 取值域，故用字面量元组而非 string[]：
 * openDocument 的 fileType 是 `keyof FileType`
 * （doc|docx|xls|xlsx|ppt|pptx|pdf），string 赋不进去。该类型由 Taro
 * 生成器产出，恰好与本白名单一致。
 */
export const OPENABLE_FORMATS = [
  'doc',
  'docx',
  'xls',
  'xlsx',
  'ppt',
  'pptx',
  'pdf',
] as const

type OpenableFormat = (typeof OPENABLE_FORMATS)[number]

function hasFormat<F extends string>(
  formats: readonly F[],
  format: string,
): format is F {
  return formats.indexOf(format as F) >= 0
}

/** 判断是不是 openDocument 认的格式；同时把类型收窄到枚举内。 */
export function canOpen(format: string): format is OpenableFormat {
  return hasFormat(OPENABLE_FORMATS, (format || '').toLowerCase())
}

/**
 * 上一次用 writeFile 写进 USER_DATA_PATH 的路径。
 * 该目录有 200MB 上限，反复下载会堆积到写不进去——每次写盘前先删掉上一个，
 * 只保留最新一份。downloadFile 落到系统临时目录，不在此列。
 *
 * **单槽保留是刻意的，不是 bug。** `saveBase64File`（artifact）与
 * `exportMarkdown`（导出的会话）共用这一个槽位，因此写新文件必然让上一个
 * 失效——先导出会话再下载附件，导出的 .md 就会被删掉；用户此时若还停在
 * `wx.openDocument` 里看着它，回到小程序就没有东西可转发了。
 * 这是拿"同时只能留一份"换 USER_DATA_PATH 的空间预算，接受该取舍。
 * 请勿把它"修"成无上限的路径列表。
 */
let lastWrittenPath = ''

function cleanupLastWritten(): void {
  if (!lastWrittenPath) return
  try {
    Taro.getFileSystemManager().unlinkSync(lastWrittenPath)
  } catch {
    // 文件可能已被系统回收，忽略
  }
  lastWrittenPath = ''
}

export function reportDownloadUrl(taskId: string, format: string): string {
  return `${API_BASE}/long_task/${encodeURIComponent(
    taskId,
  )}/report?format=${encodeURIComponent(format)}`
}

function writeFileOnce(
  filePath: string,
  data: string,
  encoding: 'base64' | 'utf8',
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    Taro.getFileSystemManager().writeFile({
      filePath,
      data,
      encoding,
      success: () => resolve(),
      fail: (err) =>
        reject(new Error((err && err.errMsg) || '文件写入失败')),
    })
  })
}

/** 把 base64 分片写成临时文件，返回文件路径。 */
export async function saveBase64File(
  filename: string,
  chunks: string[],
): Promise<string> {
  cleanupLastWritten()
  const safeName = filename || `CopiioAI_${Date.now()}.bin`
  const filePath = `${Taro.env.USER_DATA_PATH}/${safeName}`
  try {
    // writeFile 接受完整 base64 串，分片在文件系统层拼接
    await writeFileOnce(filePath, chunks.join(''), 'base64')
  } catch (err) {
    // USER_DATA_PATH 有 200MB 上限：可能被别的入口写满了。
    // 再清一次（cleanupLastWritten 只删我们记着的那个），重试一次。
    cleanupLastWritten()
    try {
      await writeFileOnce(filePath, chunks.join(''), 'base64')
    } catch {
      throw err
    }
  }
  lastWrittenPath = filePath
  return filePath
}

/** 下载长任务报告（docx/pdf），返回临时文件路径。 */
export async function downloadReport(
  taskId: string,
  format: string,
): Promise<string> {
  const token = Taro.getStorageSync(STORAGE_KEYS.wxToken)
  if (!token) {
    // 没 token 就别发请求：空 Bearer 会被后端判成 404 而非 401，
    // 于是 401 分支永不触发，登出用户看到的是"报告不存在"。
    clearAuthAndRedirect('登录已失效')
  }
  const res = await Taro.downloadFile({
    url: reportDownloadUrl(taskId, format),
    header: { Authorization: `Bearer ${token}` },
  })
  if (res.statusCode === 401) clearAuthAndRedirect()
  if (res.statusCode === 404) throw new Error('报告不存在或已过期')
  if (res.statusCode >= 400) throw new Error('报告下载失败，请稍后重试')
  return res.tempFilePath
}

/**
 * 打开；打不开的格式（或打开失败）回退到转发到聊天。
 * 返回实际走的那条路，供调用方给不同提示。
 */
export async function openOrShareFile(
  filePath: string,
  format: string,
): Promise<'opened' | 'shared'> {
  // canOpen 成功时兼当类型守卫：(format || '').toLowerCase() 的类型是 string，
  // 而 fileType 要的是字面量联合，不收 string。
  const lower = (format || '').toLowerCase()
  if (canOpen(lower)) {
    try {
      await Taro.openDocument({
        filePath,
        fileType: lower,
        showMenu: true,
      })
      return 'opened'
    } catch {
      // 落到转发——某些机型对 office 格式支持不全
    }
  }
  await Taro.shareFileMessage({ filePath })
  return 'shared'
}

/** 导出消息正文为 Markdown（对齐 web 的 CopiioAI_Chat_<时间戳>.md）。 */
export async function exportMarkdown(
  filename: string,
  content: string,
): Promise<string> {
  cleanupLastWritten()
  const filePath = `${Taro.env.USER_DATA_PATH}/${filename}`
  await writeFileOnce(filePath, content, 'utf8')
  lastWrittenPath = filePath
  return filePath
}
