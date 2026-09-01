'use client'

import { createContext, useContext, useEffect, useRef, useState, type Dispatch, type MutableRefObject, type SetStateAction } from 'react'
import { loadChatStore, persistChatToStorage } from '@/lib/chatStore'
import { loadResultsStore, restoreResultsInMessages } from '@/lib/resultsStore'
import { saveSessionMessages, authHeaders } from '@/services/api'
import { pruneResultsForPersistence } from '@/lib/results'

export interface ChatArtifact {
  artifactId: string
  format: string
  filename: string
  mimeType: string
  rowCount: number
  columnCount: number
  chunks: string[]
  complete: boolean
}

export interface AgentStep {
  round: number
  thought: string
  action: string
  paramsBrief?: string
  observationBrief?: string
  reasoningText?: string
  status: 'running' | 'done' | 'error'
}

export interface ChatStatusStep {
  id: number
  message: string
  state: 'running' | 'done'
}

export interface ChatMessage {
  id: string
  role: string
  content: string
  artifacts?: ChatArtifact[]
  taskId?: string  // long task ID for progress tracking across save/load
  resultSummary?: string  // long task report markdown preview
  patent_ids?: string[]  // hidden — carried in conversation_history for follow-up queries
  results?: {
    setId: string
    source: string
    columns: Array<{ key: string; label: string; role: string }>
    rows: Array<Record<string, unknown>>
  }
  agentSteps?: AgentStep[]  // ReAct loop timeline — collapsed after completion
  elapsedSeconds?: number   // set by agent_elapsed; drives the collapse header
}

interface ChatContextValue {
  messages: ChatMessage[]
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>
  hydrated: boolean
  input: string
  setInput: Dispatch<SetStateAction<string>>
  streaming: boolean
  setStreaming: Dispatch<SetStateAction<boolean>>
  streamingId: string | null
  setStreamingId: Dispatch<SetStateAction<string | null>>
  abortRef: MutableRefObject<AbortController | null>
  sessionId: string | null
  setSessionId: Dispatch<SetStateAction<string | null>>
  /**
   * The backend session_id already loaded/created for the current
   * conversation.  The chat page's URL-restore effect short-circuits on
   * it, so a session created mid-stream (pure-chat backfill or
   * long_task_created) can never re-trigger a restore that would clobber
   * the in-flight conversation (需求 2).
   */
  lastLoadedSidRef: MutableRefObject<string | null>
  resultsSetId: string | null
  setResultsSetId: Dispatch<SetStateAction<string | null>>
  statusSteps: ChatStatusStep[]
  setStatusSteps: Dispatch<SetStateAction<ChatStatusStep[]>>
  statusElapsed: number
  setStatusElapsed: Dispatch<SetStateAction<number>>
}

const ChatContext = createContext<ChatContextValue | null>(null)

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [hydrated, setHydrated] = useState(false)
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [streamingId, setStreamingId] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [resultsSetId, setResultsSetId] = useState<string | null>(null)
  const [statusSteps, setStatusSteps] = useState<ChatStatusStep[]>([])
  const [statusElapsed, setStatusElapsed] = useState(0)
  const abortRef = useRef<AbortController | null>(null)
  const lastLoadedSidRef = useRef<string | null>(null)

  // Two-phase hydration: the landing page (/) and /app/* routes mount
  // separate ChatProvider instances, so a client-side navigation between
  // them would otherwise discard the in-memory conversation.  Restore the
  // copy persisted by the previous provider after mount (the prerendered
  // static HTML must stay empty to avoid a hydration mismatch), and
  // re-attach full result payloads from the results store — the same
  // restore path the chat page uses for backend session loads.
  const restorePendingRef = useRef(false)
  useEffect(() => {
    const stored = loadChatStore(window.sessionStorage)
    if (stored.length > 0) {
      restorePendingRef.current = true
      setMessages(restoreResultsInMessages(stored, loadResultsStore(window.localStorage)))
    }
    // Hydration is complete once the restore pass has run — set unconditionally
    // so the landing page never flashes on async message restoration.
    setHydrated(true)
  }, [])

  // Persist every change so the next provider mount (navigation, refresh,
  // full-page fallback) can rebuild the conversation.  While a restore is
  // pending, the pre-restore empty pass must not overwrite the stored
  // copy before it has been applied.
  useEffect(() => {
    if (restorePendingRef.current) {
      if (messages.length === 0) return
      restorePendingRef.current = false
    }
    persistChatToStorage(window.sessionStorage, messages)
  }, [messages])

  // Backend session persistence (需求 2) lives at the provider level so
  // EVERY page sharing the conversation — /app/chat and /app/results —
  // saves to the backend.  Previously the save effect only existed on the
  // chat page: a follow-up asked from the results page streamed into
  // memory/sessionStorage only, so a refresh restored the stale backend
  // copy and the follow-up was lost (双 session 排查 2026-09-01).
  const pendingSaveRef = useRef(false)
  const messagesHash = JSON.stringify(messages.map(m => ({
    role: m.role,
    content: m.content,
    taskId: m.taskId,
    resultSummary: m.resultSummary,
    patent_ids: m.patent_ids,
  })))
  useEffect(() => {
    if (streaming || messages.length === 0) return
    if (!sessionId) return  // No session yet = nothing to save into
    if (pendingSaveRef.current) return
    pendingSaveRef.current = true

    const timer = setTimeout(async () => {
      try {
        const toSave = messages.map(m => ({
          role: m.role,
          content: m.content,
          ...(m.taskId ? { taskId: m.taskId } : {}),
          ...(m.resultSummary ? { resultSummary: m.resultSummary } : {}),
          ...(m.patent_ids ? { patent_ids: m.patent_ids } : {}),
          ...(m.results
            ? { results: pruneResultsForPersistence(m.results) }
            : {}),
        }))
        await saveSessionMessages(sessionId, toSave)
      } catch {
        // Non-critical — pagehide keepalive and the next change retry
      }
      pendingSaveRef.current = false
    }, 1000)

    return () => { clearTimeout(timer); pendingSaveRef.current = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- messagesHash covers the persisted fields
  }, [streaming, messagesHash, sessionId])

  // 追问后立即刷新会丢失刚保存的消息（保存有 1s 延迟）— 页面卸载前
  // (刷新/关闭) 用 fetch keepalive 尽力保存最新消息 (需求 2 补漏)。
  // Provider-level so it also covers the results page.
  useEffect(() => {
    function saveOnUnload() {
      if (!sessionId || streaming || messages.length === 0) return
      const toSave = messages.map(m => ({
        role: m.role,
        content: m.content,
        ...(m.taskId ? { taskId: m.taskId } : {}),
        ...(m.resultSummary ? { resultSummary: m.resultSummary } : {}),
        ...(m.patent_ids ? { patent_ids: m.patent_ids } : {}),
        ...(m.results
          ? { results: pruneResultsForPersistence(m.results) }
          : {}),
      }))
      // keepalive: 页面卸载期间请求仍会发出 (PUT 仅 fetch 支持,
      // sendBeacon 只支持 POST)
      void authHeaders()
        .then((headers: Record<string, string>) => {
          void fetch(`${process.env.NEXT_PUBLIC_API_BASE || 'https://api.copiioai.com'}/session/${encodeURIComponent(sessionId)}/messages`, {
            method: 'PUT',
            headers,
            body: JSON.stringify({ messages: toSave, title: '' }),
            keepalive: true,
          }).catch(() => {})
        })
        .catch(() => {})
    }
    window.addEventListener('pagehide', saveOnUnload)
    return () => window.removeEventListener('pagehide', saveOnUnload)
  }, [sessionId, streaming, messages])

  return (
    <ChatContext.Provider
      value={{
        messages,
        setMessages,
        hydrated,
        input,
        setInput,
        streaming,
        setStreaming,
        streamingId,
        setStreamingId,
        abortRef,
        sessionId,
        setSessionId,
        lastLoadedSidRef,
        resultsSetId,
        setResultsSetId,
        statusSteps,
        setStatusSteps,
        statusElapsed,
        setStatusElapsed,
      }}
    >
      {children}
    </ChatContext.Provider>
  )
}

export function useChatSession() {
  const ctx = useContext(ChatContext)
  if (!ctx) throw new Error('useChatSession must be used within ChatProvider')
  return ctx
}
