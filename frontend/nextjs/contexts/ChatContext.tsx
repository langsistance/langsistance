'use client'

import { createContext, useContext, useEffect, useRef, useState, type Dispatch, type MutableRefObject, type SetStateAction } from 'react'

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

// [DIAG] Window-lifetime marker: survives SPA navigations, dies on a full
// page load.  Lets the results page tell the two apart.
if (typeof window !== 'undefined') {
  ;(window as any).__copiioaiAlive = true
}

// [DIAG] Wrap fetch to trace the RSC flight requests Next makes during
// client-side navigation — the decisive evidence for the full-page-load
// fallback (404? blocked? redirect?).
if (typeof window !== 'undefined' && !(window as any).__copiioaiFetchDiag) {
  ;(window as any).__copiioaiFetchDiag = true
  const origFetch = window.fetch.bind(window)
  window.fetch = (async (...args: Parameters<typeof fetch>) => {
    const res = await origFetch(...args)
    try {
      const input = args[0]
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input?.url || ''
      if (url.includes('_rsc') || url.endsWith('.txt') || url.includes('index.txt')) {
        console.log('[copiioai-diag] fetch', url, '->', res.status, 'type=' + res.type, 'redirected=' + res.redirected, 'final=' + res.url)
      }
    } catch {}
    return res
  }) as typeof fetch
}

export function ChatProvider({ children }: { children: React.ReactNode }) {
  // [DIAG]
  useEffect(() => {
    try {
      const key = 'copiioai_diag'
      sessionStorage.setItem(key, (sessionStorage.getItem(key) || '') + '|P:mount')
      console.log('[copiioai-diag] provider-mount breadcrumb=', sessionStorage.getItem(key))
    } catch {}
  }, [])

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [streamingId, setStreamingId] = useState<string | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [resultsSetId, setResultsSetId] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

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
