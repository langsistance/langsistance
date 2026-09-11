import Taro from '@tarojs/taro'
import { API_BASE, STORAGE_KEYS } from '../config'
import { ChatMsg } from './chat'

/**
 * /query_stream SSE 流式客户端（M2 v1 传输：wx.request enableChunked）。
 *
 * 后端事件（core.py SSE 序列化）: 每帧 `data:{json}\n\n`, 事件含 type:
 *   token / status / error / end / long_task_created / artifact_* ...
 *   `: ping` 心跳帧忽略。
 *
 * 后续形态: 后端 /ws/chat(WebSocket) 部署后, 此处换 connectSocket 实现,
 * 对外保持同一组回调即可(协议见 WsChat 注释)。
 */
export interface StreamCallbacks {
  onToken?: (content: string) => void
  onStatus?: (content: string) => void
  onError?: (message: string) => void
  onDone?: () => void
  onEvent?: (event: Record<string, any>) => void
}

interface SseEvent {
  type?: string
  content?: string
  message?: string
  thought?: string
  [key: string]: any
}

/** ArrayBuffer → UTF-8 字符串（TextDecoder 缺失时回退 escape 方案）。 */
function decodeChunk(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf)
  // eslint-disable-next-line no-undef
  if (typeof TextDecoder !== 'undefined') {
    // eslint-disable-next-line no-undef
    return new TextDecoder('utf-8').decode(bytes)
  }
  let bin = ''
  const STEP = 8192
  for (let i = 0; i < bytes.length; i += STEP) {
    bin += String.fromCharCode.apply(
      null,
      Array.from(bytes.subarray(i, Math.min(i + STEP, bytes.length))),
    )
  }
  try {
    return decodeURIComponent(escape(bin))
  } catch {
    return bin
  }
}

type ChunkCallback = (res: { data: ArrayBuffer }) => void

/** 增量 SSE 解析器：跨 chunk 的帧/多字节字符安全。 */
class SseParser {
  private buf = ''

  push(raw: string): SseEvent[] {
    this.buf += raw
    const events: SseEvent[] = []
    for (;;) {
      const idx = this.buf.indexOf('\n\n')
      if (idx < 0) break
      const frame = this.buf.slice(0, idx)
      this.buf = this.buf.slice(idx + 2)
      const ev = this.parseFrame(frame)
      if (ev) events.push(ev)
    }
    return events
  }

  private parseFrame(frame: string): SseEvent | null {
    const lines = frame.split('\n')
    let data = ''
    for (const line of lines) {
      if (line.startsWith(':')) continue // 心跳/注释
      if (line.startsWith('data:')) {
        data += line.slice(5).trim()
      }
    }
    if (!data) return null
    try {
      const parsed = JSON.parse(data)
      // 正文 token 与其余事件的线路格式不同：
      //   token           → 裸 JSON 字符串，如 data:"##"
      //                      （后端 core.py 对缓冲后的正文做 json.dumps(字符串)）
      //   status/step/... → 带 type 的对象
      // 因此字符串载荷要归一成 {type:'token', content} 再交给 handleEvent，
      // 否则会落进 default 分支被静默丢弃（正文永远为空）。
      if (typeof parsed === 'string') {
        return { type: 'token', content: parsed }
      }
      return parsed as SseEvent
    } catch {
      return null
    }
  }
}

/** 发起一次流式问答。resolve 于收到 end；reject 于错误事件/网络失败。 */
export function streamQuery(
  query: string,
  conversationHistory: ChatMsg[],
  cb: StreamCallbacks,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const parser = new SseParser()
    let finished = false
    const token = Taro.getStorageSync(STORAGE_KEYS.wxToken)
    const queryId = `mini_${Date.now().toString(36)}_${Math.floor(
      Math.random() * 1e6,
    ).toString(36)}`

    const finish = (err?: Error) => {
      if (finished) return
      finished = true
      if (err) reject(err)
      else resolve()
    }

    const onChunk: ChunkCallback = (res) => {
      try {
        const raw = decodeChunk(res.data)
        for (const ev of parser.push(raw)) {
          handleEvent(ev, cb, finish)
        }
      } catch (e) {
        finish(e instanceof Error ? e : new Error('响应解析失败'))
      }
    }

    const req = Taro.request({
      url: `${API_BASE}/query_stream`,
      method: 'POST',
      enableChunked: true,
      timeout: 120000,
      header: {
        'content-type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      data: {
        query,
        query_id: queryId,
        tts_enabled: false,
        tool_data: '',
        push_filter: null,
        conversation_history: conversationHistory,
      },
    })

    // enableChunked 的流式监听必须挂在本次请求的 RequestTask 上：
    // onChunkReceived/offChunkReceived 是 RequestTask 的方法（Taro.request()
    // 的返回值），Taro 命名空间上没有这两个 API。
    req.onChunkReceived(onChunk)

    const timer = setTimeout(() => {
      req.offChunkReceived(onChunk)
      finish(new Error('请求超时，请重试'))
    }, 150000)

    req
      .then((resp) => {
        // 流结束后服务器已发 end 事件即 resolve；此处兜底：
        // 非 200 且无 error 帧时给可读错误
        if (resp.statusCode >= 400) {
          const body: any = (resp as any).data
          const detail = body && (body.detail || body.message)
          finish(
            new Error(
              typeof detail === 'string' ? detail : `请求失败(${resp.statusCode})`,
            ),
          )
        } else {
          finish()
        }
      })
      .catch((err: any) => {
        finish(
          new Error(
            (err && err.errMsg) || '网络错误，请检查连接后重试',
          ),
        )
      })
      .finally(() => {
        clearTimeout(timer)
        req.offChunkReceived(onChunk)
      })
  })
}

function handleEvent(
  ev: SseEvent,
  cb: StreamCallbacks,
  finish: (err?: Error) => void,
) {
  switch (ev.type) {
    case 'token':
      cb.onToken?.(ev.content || '')
      break
    case 'status':
      // 状态帧的载荷字段是 message（不是 content）——读 content 会恒为空串，
      // 等待期界面上什么都看不到。
      cb.onStatus?.(ev.message || ev.content || '')
      break
    case 'step':
      // 步骤帧是等待期信息量最大的一条：第 N 步 · 正在调用「工具名」
      if (ev.thought) cb.onStatus?.(String(ev.thought))
      break
    case 'error': {
      const msg = (ev.message || ev.content || '服务出错，请重试') as string
      cb.onError?.(msg)
      finish(new Error(msg))
      break
    }
    case 'end':
      cb.onDone?.()
      finish()
      break
    default:
      cb.onEvent?.(ev)
  }
}

/**
 * 预留：/ws/chat(WebSocket) 形态协议（后端实现后启用, 回调形状与上面一致）。
 *   连接: ws(s)://<host>/ws/chat
 *   客户端帧: {"type":"query","query_id","query","history":[...]}
 *   服务端帧: {"type":"token"|"status"|"patents"|"end"|"error", ...}
 */
export function wsUrl(): string {
  const base = (process.env.TARO_APP_WS_BASE ||
    API_BASE.replace(/^http/, 'ws')) as string
  return `${base.replace(/\/$/, '')}/ws/chat`
}
