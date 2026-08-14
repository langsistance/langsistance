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
}

interface ChatContextValue {
  messages: ChatMessage[]
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>
  input: string
  setInput: Dispatch<SetStateAction<string>>
  streaming: boolean
  setStreaming: Dispatch<SetStateAction<boolean>>
  streamingId: string | null
  setStreamingId: Dispatch<SetStateAction<string | null>>
  abortRef: MutableRefObject<AbortController | null>
  sessionId: string | null
  setSessionId: Dispatch<SetStateAction<string | null>>
  resultsSetId: string | null
  setResultsSetId: Dispatch<SetStateAction<string | null>>
}

const ChatContext = createContext<ChatContextValue | null>(null)

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [streamingId, setStreamingId] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [resultsSetId, setResultsSetId] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

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
        input,
        setInput,
        streaming,
        setStreaming,
        streamingId,
        setStreamingId,
        abortRef,
        sessionId,
        setSessionId,
        resultsSetId,
        setResultsSetId,
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
