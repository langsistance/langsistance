import { useState, useRef, useEffect } from 'react'
import { queryStream } from '../services/api'

function genId() {
  return Math.random().toString(36).slice(2)
}

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const abortRef = useRef(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send() {
    const text = input.trim()
    if (!text || streaming) return
    setInput('')

    const queryId = genId()
    setMessages((m) => [...m, { role: 'user', content: text, id: genId() }])

    const assistantId = genId()
    setMessages((m) => [...m, { role: 'assistant', content: '', id: assistantId }])
    setStreaming(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const body = await queryStream(text, queryId, controller.signal)
      const reader = body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const raw = line.slice(5).trim()
          if (raw === '[DONE]') continue
          try {
            const evt = JSON.parse(raw)
            const token = evt.content ?? evt.token ?? evt.answer ?? ''
            if (token) {
              setMessages((m) =>
                m.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, content: msg.content + token }
                    : msg
                )
              )
            }
          } catch {
            // non-JSON line, ignore
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantId
              ? { ...msg, content: '请求失败，请重试。' }
              : msg
          )
        )
      }
    } finally {
      setStreaming(false)
      abortRef.current = null
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  function abort() {
    abortRef.current?.abort()
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-slate-800">
        <h2 className="text-sm font-semibold text-slate-200">对话</h2>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3">
        {messages.length === 0 && (
          <p className="text-slate-500 text-sm text-center mt-8">发送消息开始对话</p>
        )}
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`max-w-[80%] rounded-xl px-3 py-2 text-sm whitespace-pre-wrap break-words ${
              msg.role === 'user'
                ? 'self-start bg-slate-700 text-slate-200'
                : 'self-end bg-teal-600 text-white'
            }`}
          >
            {msg.content || (msg.role === 'assistant' && streaming ? '▋' : '')}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="px-4 py-3 border-t border-slate-800 flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入消息..."
          rows={1}
          className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 resize-none focus:outline-none focus:border-teal-500"
        />
        {streaming ? (
          <button
            onClick={abort}
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-sm text-white rounded-lg"
          >
            停止
          </button>
        ) : (
          <button
            onClick={send}
            disabled={!input.trim()}
            className="px-4 py-2 bg-teal-600 hover:bg-teal-500 disabled:opacity-40 text-sm text-white rounded-lg"
          >
            发送
          </button>
        )}
      </div>
    </div>
  )
}
