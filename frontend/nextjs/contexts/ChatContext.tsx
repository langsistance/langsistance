'use client'

import { createContext, useContext, useEffect, useRef, useState, type Dispatch, type MutableRefObject, type SetStateAction } from 'react'
import { loadChatStore, persistChatToStorage } from '@/lib/chatStore'
import { loadResultsStore, restoreResultsInMessages } from '@/lib/resultsStore'

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
