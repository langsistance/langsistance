import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Button,
  RichText,
  ScrollView,
  Text,
  Textarea,
  View,
} from '@tarojs/components'
import Taro, { useRouter } from '@tarojs/taro'
import {
  ChatMsg,
  createSession,
  extractPatentIds,
  fetchSession,
  saveMessages,
} from '../../services/chat'
import { streamQuery } from '../../services/chatStream'
import { errorText } from '../../services/api'
import { markdownToHtml } from '../../utils/markdown'
import './index.scss'

interface MsgView {
  role: 'user' | 'assistant'
  content: string
  html?: string
  patents?: string[]
  streaming?: boolean
}

/**
 * M2 对话页：SSE 流式对话（/query_stream, enableChunked）
 * + 专利号结果卡（过渡版：提取自助手终稿文本；完整面板随结果事件通道落地）。
 * 富文本经 <RichText> 渲染（weapp 正确原语；内容来自自有后端，用户输入恒为纯文本）。
 */
export default function ChatPage() {
  const router = useRouter()
  const sessionIdRef = useRef(router.params.session_id || '')
  const assistantRef = useRef('') // 最新一轮助手全文（落库用，防闭包过期）
  const [msgs, setMsgs] = useState<MsgView[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [anchor, setAnchor] = useState('')

  const scrollToBottom = () => setAnchor(`msg-${Date.now()}`)

  // 载入历史会话
  useEffect(() => {
    const sid = sessionIdRef.current
    if (!sid) return
    fetchSession(sid)
      .then((detail) => {
        const history: MsgView[] = (detail.messages || [])
          .filter((m) => m.role === 'user' || m.role === 'assistant')
          .map((m: ChatMsg) => ({
            role: m.role as 'user' | 'assistant',
            content: m.content || '',
          }))
        setMsgs(history)
        scrollToBottom()
      })
      .catch((err) => setError(errorText(err, '历史会话加载失败')))
  }, [])

  const appendToken = useCallback((chunk: string) => {
    assistantRef.current += chunk
    setMsgs((prev) => {
      const next = prev.slice()
      const last = next[next.length - 1]
      if (last && last.role === 'assistant') {
        next[next.length - 1] = { ...last, content: assistantRef.current }
      }
      return next
    })
  }, [])

  const startAssistant = useCallback(() => {
    assistantRef.current = ''
    setMsgs((prev) => [
      ...prev,
      { role: 'assistant', content: '', streaming: true },
    ])
  }, [])

  const finalizeAssistant = useCallback(() => {
    setMsgs((prev) => {
      const next = prev.slice()
      const last = next[next.length - 1]
      if (last && last.role === 'assistant') {
        next[next.length - 1] = {
          ...last,
          streaming: false,
          patents: extractPatentIds(last.content),
        }
      }
      return next
    })
  }, [])

  async function send() {
    const text = input.trim()
    if (!text || sending) return
    setInput('')
    setSending(true)
    setStatus('连接中…')
    setError('')
    try {
      const history: ChatMsg[] = msgs.map((m) => ({
        role: m.role,
        content: m.content,
      }))
      const userMsg: ChatMsg = { role: 'user', content: text }

      // 首条消息建会话（scene 1 = 专利检索默认场景）
      let sid = sessionIdRef.current
      if (!sid) {
        sid = await createSession(text, [...history, userMsg])
        sessionIdRef.current = sid
      }
      setMsgs((prev) => [...prev, { role: 'user', content: text }])
      scrollToBottom()

      startAssistant()
      let completed = false
      try {
        await streamQuery(text, history, {
          onStatus: (s) => {
            if (!completed) setStatus(s)
          },
          onToken: (chunk) => {
            if (!completed) {
              setStatus('')
              appendToken(chunk)
            }
          },
          onError: (message) => setError(message),
        })
      } finally {
        completed = true
        finalizeAssistant()
      }
      // 终稿落库（web 同款持久化）
      const assistantMsg: ChatMsg = {
        role: 'assistant',
        content: assistantRef.current,
      }
      await saveMessages(sid, [...history, userMsg, assistantMsg])
    } catch (err) {
      setError(errorText(err))
    } finally {
      setSending(false)
      setStatus('')
      scrollToBottom()
    }
  }

  function copyPatent(pid: string) {
    Taro.setClipboardData({ data: pid })
  }

  return (
    <View className='chat'>
      <ScrollView
        className='chat-scroll'
        scrollY
        scrollIntoView={anchor}
        scrollWithAnimation
      >
        <View className='chat-list'>
          {msgs.length === 0 ? (
            <View className='chat-welcome'>
              <Text className='chat-welcome-title'>专利智能对话</Text>
              <Text className='chat-welcome-sub text-muted'>
                描述您的产品、技术或专利号，例如：
              </Text>
              <Text className='chat-welcome-example text-muted'>
                “查一下可折叠桌子相关的专利”
              </Text>
            </View>
          ) : null}

          {msgs.map((m, i) => (
            <View
              key={`${m.role}-${i}`}
              className={`chat-msg chat-msg-${m.role}`}
            >
              {m.role === 'assistant' && m.content ? (
                <View className='chat-msg-body'>
                  <RichText nodes={markdownToHtml(m.content)} />
                </View>
              ) : (
                <View className='chat-msg-body'>
                  <Text className='chat-msg-text'>
                    {m.content || (m.streaming ? '…' : '')}
                  </Text>
                </View>
              )}
              {m.role === 'assistant' && m.patents && m.patents.length > 0 ? (
                <View className='chat-msg-patents'>
                  {m.patents.map((pid) => (
                    <View
                      key={pid}
                      className='chat-patent'
                      onClick={() => copyPatent(pid)}
                    >
                      <Text className='chat-patent-id'>{pid}</Text>
                      <Text className='chat-patent-copy'>复制</Text>
                    </View>
                  ))}
                </View>
              ) : null}
            </View>
          ))}

          {status ? (
            <View className='chat-status'>
              <Text className='text-muted'>{status}</Text>
            </View>
          ) : null}
          {error ? (
            <View className='chat-err'>
              <Text>{error}</Text>
            </View>
          ) : null}
        </View>
      </ScrollView>

      <View className='chat-inputbar'>
        <Textarea
          className='chat-textarea'
          value={input}
          maxlength={4000}
          autoHeight
          placeholder='输入问题…'
          placeholderClass='chat-placeholder'
          onInput={(e) => setInput(e.detail.value)}
          disabled={sending}
        />
        <Button
          className='chat-send'
          disabled={sending || !input.trim()}
          onClick={send}
        >
          {sending ? '…' : '发送'}
        </Button>
      </View>
    </View>
  )
}
