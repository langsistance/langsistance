import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Button,
  RichText,
  ScrollView,
  Text,
  Textarea,
  View,
} from '@tarojs/components'
import Taro, {
  useDidHide,
  useDidShow,
  useShareAppMessage,
  useShareTimeline,
} from '@tarojs/taro'
import {
  ChatMsg,
  createSession,
  extractPatentIds,
  fetchSession,
  saveMessages,
} from '../../services/chat'
import { CompletedArtifact, streamQuery } from '../../services/chatStream'
import { errorText } from '../../services/api'
import { isLoggedIn } from '../../services/auth'
import {
  archiveSession,
  fetchSessions,
  MAX_TITLE_LEN,
  renameSession,
  SessionItem,
} from '../../services/sessions'
import {
  MAX_POLL_FAILURES,
  POLL_INTERVAL_MS,
  isTerminal,
  pollTasks,
  retryTask,
  TaskState,
} from '../../services/longTask'
import { uploadQuery } from '../../services/upload'
import AttachmentBar from '../../components/AttachmentBar'
import LongTaskCard from '../../components/LongTaskCard'
import NavBar from '../../components/NavBar'
import SessionDrawer from '../../components/SessionDrawer'
import RenameModal from '../../components/RenameModal'
import PrivacyPopup from '../../components/PrivacyPopup'
import { pickFile, PickedFile } from '../../services/upload'
import {
  downloadReport,
  exportMarkdown,
  openOrShareFile,
  saveBase64File,
} from '../../services/download'
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
  /** 本轮回答附带的可下载工件（CSV/XLSX…），收齐后一次性挂上 */
  artifacts?: CompletedArtifact[]
  streaming?: boolean
  /** 长任务状态（上传分支专用）。不走 web 的标记编码 + 正则反解——
      小程序的消息本来就是结构化对象。 */
  task?: TaskState
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

  // msgs 的镜像。轮询回调闭包捕获的是某一次渲染的 msgs，而 startPolling 在
  // send() 里被同步调用——那时 setMsgs 还没提交，闭包里看不到刚挂上的 task，
  // 首次 tick 会得到空 id 列表并立刻自停，卡片就此冻结到下次进入页面。
  const msgsRef = useRef<MsgView[]>([])

  // 长任务轮询：taskId → 连续失败次数
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollFailRef = useRef<Record<string, number>>({})

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

  function stopPolling() {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }

  /** 收集当前仍未结束的任务号。读 msgsRef 而非 msgs —— 见 msgsRef 的注释。 */
  function activeTaskIds(): string[] {
    return msgsRef.current
      .filter((m) => m.task && !isTerminal(m.task.status))
      .map((m) => m.task!.taskId)
  }

  function applyTaskStates(states: Record<string, TaskState>) {
    setMsgs((prev) =>
      prev.map((m) =>
        m.task && states[m.task.taskId]
          ? { ...m, task: states[m.task.taskId] }
          : m,
      ),
    )
  }

  function startPolling() {
    if (pollTimerRef.current) return
    pollTimerRef.current = setInterval(async () => {
      const ids = activeTaskIds()
      if (ids.length === 0) {
        stopPolling()
        return
      }
      try {
        const states = await pollTasks(ids)
        pollFailRef.current = {}
        applyTaskStates(states)
        if (ids.every((id) => states[id] && isTerminal(states[id].status))) {
          stopPolling()
        }
      } catch {
        // 连续失败才判死，避免一次抖动就把卡片钉在"状态获取失败"
        const ids2 = activeTaskIds()
        for (const id of ids2) {
          pollFailRef.current[id] = (pollFailRef.current[id] || 0) + 1
        }
        const dead = ids2.filter(
          (id) => pollFailRef.current[id] >= MAX_POLL_FAILURES,
        )
        if (dead.length) {
          setMsgs((prev) =>
            prev.map((m) =>
              m.task && dead.indexOf(m.task.taskId) >= 0
                ? { ...m, task: { ...m.task, status: 'unknown' as const } }
                : m,
            ),
          )
        }
        if (
          ids2.every((id) => (pollFailRef.current[id] || 0) >= MAX_POLL_FAILURES)
        ) {
          stopPolling()
        }
      }
    }, POLL_INTERVAL_MS)
  }

  // 离开页面停止轮询；回来对未完成任务恢复
  useDidHide(() => stopPolling())
  useDidShow(() => {
    if (activeTaskIds().length > 0) startPolling()
  })

  // 分享：只做转发卡片，path 指向首页。不做分享特定会话——会话有归属校验，
  // 转发出去对方只会看到「会话不存在」。
  useShareAppMessage(() => ({
    title: sessionTitle ? `CopiioAI：${sessionTitle}` : 'CopiioAI 专利助手',
    path: '/pages/chat/index',
  }))

  useShareTimeline(() => ({
    title: sessionTitle ? `CopiioAI：${sessionTitle}` : 'CopiioAI 专利助手',
    query: '',
  }))

  useEffect(() => {
    Taro.showShareMenu({
      showShareItems: ['shareAppMessage', 'shareTimeline'],
    })
  }, [])

  // 卸载时清理：停计时器与轮询 + 结算可能悬着的隐私授权等待
  useEffect(
    () => () => {
      stopTimer()
      stopPolling()
      abandonPrivacy()
    },
    [],
  )

  // 订阅隐私浮层显隐（App 级监听在 app.ts 注册）
  useEffect(() => subscribePrivacy(setPrivacyVisible), [])

  // 把已提交的 msgs 同步进 msgsRef，供轮询回调读取最新任务列表
  useEffect(() => {
    msgsRef.current = msgs
  }, [msgs])

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

  // 工件收齐回调（artifact_end 后一次性到达）。appendToken / finalizeAssistant
  // 都是展开旧对象换新，artifacts 会被自动带过去；但这两处都在流中途发生，
  // 必须用函数式 setMsgs 拿最新一条，否则会把并发写入的正文覆盖回去。
  const onArtifactsReady = useCallback((items: CompletedArtifact[]) => {
    if (items.length === 0) return
    setMsgs((prev) => {
      const next = prev.slice()
      const last = next[next.length - 1]
      if (last && last.role === 'assistant') {
        next[next.length - 1] = {
          ...last,
          artifacts: [...(last.artifacts || []), ...items],
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

      // 本轮是否走上传分支。发送开始时快照：上传期间若用户又点了 ＋，
      // 不应影响已经开始的本轮。
      const file = attachedFile

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

      if (file) {
        // ── 上传分支：单文件 → 长任务 ──
        // conversation_history 必须包含本轮新提问：后端会把它直接写成会话的
        // messages（api_routes/core.py:1041），不含新提问则该轮不落库。
        const outcome = await uploadQuery(
          file,
          text,
          [...history, userMsg],
          newQueryId(),
          sid,
        )
        // 后端可能复用/新建了别的 session_id，以它为准
        if (outcome.sessionId) sessionIdRef.current = outcome.sessionId
        setMsgs((prev) => {
          const next = prev.slice()
          const last = next[next.length - 1]
          if (last && last.role === 'assistant') {
            next[next.length - 1] = {
              ...last,
              streaming: false,
              task: {
                taskId: outcome.taskId,
                status: 'queued',
                phase: '',
                progress: 0,
                step: '',
                reportFiles: [],
                error: '',
              },
            }
          }
          return next
        })
        setAttachedFile(null)
        startPolling()
        // ⚠️ 此处**不调用 saveMessages**：后端已在本次请求内把
        // conversation_history 写进会话，随后还会追加 created/completed/
        // failed 消息。小程序若再 PUT /messages 会整体重写数组，抹掉它们。
      } else {
        // ── 普通问答分支 ──
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
            onArtifactsReady,
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
      }
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

  function newQueryId(): string {
    return `mini_${Date.now().toString(36)}_${Math.floor(
      Math.random() * 1e6,
    ).toString(36)}`
  }

  async function handleRetryTask(taskId: string) {
    try {
      const newId = await retryTask(taskId)
      setMsgs((prev) =>
        prev.map((m) =>
          m.task && m.task.taskId === taskId
            ? {
                ...m,
                task: {
                  taskId: newId,
                  status: 'queued' as const,
                  phase: '',
                  progress: 0,
                  step: '',
                  reportFiles: [],
                  error: '',
                },
              }
            : m,
        ),
      )
      pollFailRef.current = {}
      startPolling()
    } catch (err) {
      Taro.showToast({ title: errorText(err, '重试失败'), icon: 'none' })
    }
  }

  async function handleDownloadReport(taskId: string, format: string) {
    try {
      Taro.showLoading({ title: '正在下载…' })
      const path = await downloadReport(taskId, format)
      Taro.hideLoading()
      const how = await openOrShareFile(path, format)
      if (how === 'shared') {
        Taro.showToast({ title: '已转发到聊天', icon: 'none' })
      }
    } catch (err) {
      Taro.hideLoading()
      Taro.showToast({ title: errorText(err, '下载失败'), icon: 'none' })
    }
  }

  async function handleDownloadArtifact(a: CompletedArtifact) {
    try {
      Taro.showLoading({ title: '正在保存…' })
      const path = await saveBase64File(a.filename, a.chunks)
      Taro.hideLoading()
      const how = await openOrShareFile(path, a.format)
      if (how === 'shared') {
        Taro.showToast({ title: '已转发到聊天', icon: 'none' })
      }
    } catch (err) {
      Taro.hideLoading()
      Taro.showToast({ title: errorText(err, '保存失败'), icon: 'none' })
    }
  }

  async function handleExportMarkdown(m: MsgView) {
    try {
      const ts = new Date()
        .toISOString()
        .replace(/[:.]/g, '-')
        .slice(0, -5)
      const path = await exportMarkdown(`CopiioAI_Chat_${ts}.md`, m.content)
      await openOrShareFile(path, 'md')
      Taro.showToast({ title: '已转发到聊天', icon: 'none' })
    } catch (err) {
      Taro.showToast({ title: errorText(err, '下载失败'), icon: 'none' })
    }
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
              {m.role === 'assistant' &&
              (m.artifacts || []).filter((a) => a.format !== 'json').length > 0 ? (
                <View className='chat-msg-artifacts'>
                  {(m.artifacts || [])
                    .filter((a) => a.format !== 'json')
                    .map((a) => (
                      <View
                        key={a.artifactId}
                        className='chat-artifact'
                        onClick={() => handleDownloadArtifact(a)}
                      >
                        <Text className='chat-artifact-badge'>
                          {a.format.toUpperCase()}
                        </Text>
                        <Text className='chat-artifact-label'>
                          {a.format === 'csv' ? '下载 CSV' : '下载 Excel'}
                        </Text>
                      </View>
                    ))}
                </View>
              ) : null}

              {m.role === 'assistant' && m.content ? (
                <View
                  className='chat-artifact chat-artifact-plain'
                  onClick={() => handleExportMarkdown(m)}
                >
                  <Text className='chat-artifact-badge'>MD</Text>
                  <Text className='chat-artifact-label'>下载原文</Text>
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
              {m.role === 'assistant' && m.task ? (
                <LongTaskCard
                  task={m.task}
                  onRetry={() => handleRetryTask(m.task!.taskId)}
                  onDownload={(format) =>
                    handleDownloadReport(m.task!.taskId, format)
                  }
                />
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
        onRemove={() => setAttachedFile(null)}
      />

      <View className='chat-inputbar'>
        <View
          className={`chat-attach${sending ? ' chat-attach-busy' : ''}`}
          onClick={sending ? undefined : addAttachment}
        >
          <Text className='chat-attach-icon'>＋</Text>
        </View>
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
          {sending ? (
            <Text className='chat-send-icon'>…</Text>
          ) : (
            <View className='chat-send-arrow'>
              <View className='chat-send-stem' />
              <View className='chat-send-head chat-send-head-left' />
              <View className='chat-send-head chat-send-head-right' />
            </View>
          )}
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
        // buttonId 由 resolvePrivacy 的默认参数提供（PRIVACY_AGREE_BUTTON_ID），
        // 必须与 PrivacyPopup 按钮的 id 一致，微信才会放行
        onAgree={() => resolvePrivacy(true)}
        onDecline={() => resolvePrivacy(false)}
      />
    </View>
  )
}
