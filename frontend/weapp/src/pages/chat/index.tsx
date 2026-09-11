import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Button,
  RichText,
  ScrollView,
  Text,
  Textarea,
  View,
} from '@tarojs/components'
import Taro, { useDidShow } from '@tarojs/taro'
import {
  ChatMsg,
  createSession,
  extractPatentIds,
  fetchSession,
  saveMessages,
} from '../../services/chat'
import { streamQuery } from '../../services/chatStream'
import { errorText } from '../../services/api'
import { isLoggedIn } from '../../services/auth'
import {
  archiveSession,
  fetchSessions,
  MAX_TITLE_LEN,
  renameSession,
  SessionItem,
} from '../../services/sessions'
import AttachmentBar from '../../components/AttachmentBar'
import NavBar from '../../components/NavBar'
import SessionDrawer from '../../components/SessionDrawer'
import RenameModal from '../../components/RenameModal'
import PrivacyPopup from '../../components/PrivacyPopup'
import { pickFile, PickedFile } from '../../services/upload'
import {
  abandonPrivacy,
  resolvePrivacy,
  subscribePrivacy,
} from '../../services/privacy'
import { parseMarkdown } from '../../utils/markdown'
import './index.scss'

interface MsgView {
  role: 'user' | 'assistant'
  content: string
  html?: string
  patents?: string[]
  streaming?: boolean
}

/**
 * 形态一：首页即对话页（DeepSeek App 式）。
 * 左上角 ☰ 呼出抽屉收历史，抽屉内可新建/切换/重命名/删除。
 * 富文本经 <RichText> 渲染（weapp 正确原语；内容来自自有后端，用户输入恒为纯文本）。
 */
export default function ChatPage() {
  const sessionIdRef = useRef('')
  const assistantRef = useRef('') // 最新一轮助手全文（落库用，防闭包过期）
  const [msgs, setMsgs] = useState<MsgView[]>([])
  const [sessionTitle, setSessionTitle] = useState('')
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [anchor, setAnchor] = useState('')

  // 附件（本轮待上传文件，Task 9 接入发送分支后才真正上传）
  const [attachedFile, setAttachedFile] = useState<PickedFile | null>(null)

  // 隐私授权浮层
  const [privacyVisible, setPrivacyVisible] = useState(false)

  // 抽屉
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [sessions, setSessions] = useState<SessionItem[]>([])
  const [listLoading, setListLoading] = useState(false)
  const [listError, setListError] = useState('')

  // 重命名弹窗
  const [renameTarget, setRenameTarget] = useState<SessionItem | null>(null)
  const [renaming, setRenaming] = useState(false)
  const [renameError, setRenameError] = useState('')

  const scrollToBottom = () => setAnchor(`msg-${Date.now()}`)

  // 处理耗时（本地计时）：等待期状态后面显示秒数。
  // 后端的 agent_elapsed 只在结束时推一次，等待期看不到跳动。
  // 语义是「当前这一步花了多久」——每轮发送、以及每次状态变化都从 0 起算。
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const timerT0Ref = useRef(0)
  const statusRef = useRef('')

  function restartTimer() {
    setElapsed(0)
    timerT0Ref.current = Date.now()
    if (timerRef.current) clearInterval(timerRef.current)
    timerRef.current = setInterval(
      () => setElapsed(Math.floor((Date.now() - timerT0Ref.current) / 1000)),
      1000,
    )
  }

  function stopTimer() {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  // 卸载时清理：停计时器 + 结算可能悬着的隐私授权等待
  useEffect(
    () => () => {
      stopTimer()
      abandonPrivacy()
    },
    [],
  )

  // 订阅隐私浮层显隐（App 级监听在 app.ts 注册）
  useEffect(() => subscribePrivacy(setPrivacyVisible), [])

  // 首页必须自己把门（登录态检查从已删除的会话列表页搬来）
  useDidShow(() => {
    if (!isLoggedIn()) {
      Taro.navigateTo({ url: '/pages/login/index' })
    }
  })

  const loadSessions = useCallback(async () => {
    setListLoading(true)
    setListError('')
    try {
      setSessions(await fetchSessions())
    } catch (err) {
      setListError(errorText(err, '会话列表加载失败'))
    } finally {
      setListLoading(false)
    }
  }, [])

  function openDrawer() {
    setDrawerOpen(true)
    loadSessions()
  }

  function resetToNewChat() {
    sessionIdRef.current = ''
    assistantRef.current = ''
    setMsgs([])
    setSessionTitle('')
    setError('')
    setStatus('')
  }

  /** ＋ 新对话：只重置本地状态，不立刻建会话（否则每点一次留一条空会话）。 */
  function newChat() {
    resetToNewChat()
    setDrawerOpen(false)
  }

  async function selectSession(sessionId: string) {
    setDrawerOpen(false)
    if (sessionId === sessionIdRef.current) return

    sessionIdRef.current = sessionId
    setError('')
    setMsgs([])
    try {
      const detail = await fetchSession(sessionId)
      const history: MsgView[] = (detail.messages || [])
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .map((m: ChatMsg) => ({
          role: m.role as 'user' | 'assistant',
          content: m.content || '',
        }))
      setMsgs(history)
      setSessionTitle(detail.title || '')
      scrollToBottom()
    } catch (err) {
      // 顺序要紧：resetToNewChat() 内部会 setError('')，
      // 必须先重置再设错误，否则错误提示会被同批 state 更新覆盖掉。
      resetToNewChat()
      setError(errorText(err, '历史会话加载失败'))
    }
  }

  async function confirmRename(title: string) {
    const target = renameTarget
    if (!target) return
    setRenaming(true)
    setRenameError('')
    try {
      await renameSession(target.session_id, title)
      setSessions((prev) =>
        prev.map((s) =>
          s.session_id === target.session_id ? { ...s, title } : s,
        ),
      )
      if (target.session_id === sessionIdRef.current) setSessionTitle(title)
      setRenameTarget(null)
    } catch (err) {
      setRenameError(errorText(err, '重命名失败，请重试'))
    } finally {
      setRenaming(false)
    }
  }

  function removeSession(session: SessionItem) {
    Taro.showModal({
      title: '删除对话',
      content: `确定删除「${session.title || '未命名对话'}」吗？`,
      confirmText: '删除',
      confirmColor: '#d32f2f',
      success: async (res) => {
        if (!res.confirm) return
        try {
          await archiveSession(session.session_id)
          setSessions((prev) =>
            prev.filter((s) => s.session_id !== session.session_id),
          )
          // 删的正好是当前会话 → 回空态，避免停在已归档会话上
          if (session.session_id === sessionIdRef.current) resetToNewChat()
        } catch (err) {
          Taro.showToast({
            title: errorText(err, '删除失败，请重试'),
            icon: 'none',
          })
        }
      },
    })
  }

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
    statusRef.current = ''
    restartTimer()
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
        setSessionTitle(text.slice(0, MAX_TITLE_LEN))
      }
      setMsgs((prev) => [...prev, { role: 'user', content: text }])
      scrollToBottom()

      startAssistant()
      let completed = false
      try {
        await streamQuery(text, history, {
          onStatus: (s) => {
            if (completed) return
            // 状态文字变化 → 计时归零（重复的同一条状态不重置，避免抖动）
            if (s && s !== statusRef.current) {
              statusRef.current = s
              restartTimer()
            }
            setStatus(s)
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
      stopTimer()
      setSending(false)
      setStatus('')
      scrollToBottom()
    }
  }

  async function addAttachment() {
    const picked = await pickFile()
    if (picked) setAttachedFile(picked)
  }

  function copyPatent(pid: string) {
    Taro.setClipboardData({ data: pid })
  }

  return (
    <View className='chat'>
      <NavBar title={sessionTitle || '新对话'} onMenuClick={openDrawer} />

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
              {m.role === 'user' ? (
                <View className='chat-msg-body'>
                  <Text className='chat-msg-text'>{m.content}</Text>
                </View>
              ) : m.content ? (
                <View className='chat-msg-body'>
                  {parseMarkdown(m.content).map((seg, si) =>
                    seg.kind === 'html' ? (
                      <View key={si} className='chat-md-block'>
                        <RichText nodes={seg.html} />
                      </View>
                    ) : (
                      <View key={si} className='chat-table'>
                        <View className='chat-table-row chat-table-head'>
                          {seg.table.headers.map((h, hi) => (
                            <Text key={hi} className='chat-table-cell'>
                              {h}
                            </Text>
                          ))}
                        </View>
                        {seg.table.rows.map((row, ri) => (
                          <View key={ri} className='chat-table-row'>
                            {row.map((cell, ci) => (
                              <Text key={ci} className='chat-table-cell'>
                                {cell}
                              </Text>
                            ))}
                          </View>
                        ))}
                      </View>
                    ),
                  )}
                </View>
              ) : null}
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

          {sending ? (
            <View className='chat-status'>
              <Text className='chat-status-text'>{status || '处理中…'}</Text>
              <Text className='chat-status-time'>{elapsed}s</Text>
            </View>
          ) : null}
          {error ? (
            <View className='chat-err'>
              <Text>{error}</Text>
            </View>
          ) : null}
        </View>
      </ScrollView>

      <AttachmentBar
        file={attachedFile}
        busy={sending}
        onAdd={addAttachment}
        onRemove={() => setAttachedFile(null)}
      />

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

      <SessionDrawer
        visible={drawerOpen}
        sessions={sessions}
        currentSessionId={sessionIdRef.current}
        loading={listLoading}
        error={listError}
        onClose={() => setDrawerOpen(false)}
        onSelect={selectSession}
        onNew={newChat}
        onRename={(s) => {
          setRenameError('')
          setRenameTarget(s)
        }}
        onDelete={removeSession}
      />

      <RenameModal
        visible={renameTarget !== null}
        initialTitle={renameTarget?.title || ''}
        busy={renaming}
        error={renameError}
        onCancel={() => setRenameTarget(null)}
        onConfirm={confirmRename}
      />

      <PrivacyPopup
        visible={privacyVisible}
        onAgree={() => resolvePrivacy(true)}
        onDecline={() => resolvePrivacy(false)}
      />
    </View>
  )
}
