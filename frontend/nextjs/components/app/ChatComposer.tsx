'use client'

import { useEffect, useRef } from 'react'
import { useI18n } from '@/lib/app-i18n'

export type ChatComposerProps = {
  input: string
  setInput: (value: string) => void
  streaming: boolean
  send: (files?: File[], presetText?: string) => Promise<void>
  abort: () => void
  selectedFiles: File[]
  addFiles: (files: FileList | File[]) => void
  removeFile: (index: number) => void
  setIsDragOver: (value: boolean) => void
}

export default function ChatComposer({
  input,
  setInput,
  streaming,
  send,
  abort,
  selectedFiles,
  addFiles,
  removeFile,
  setIsDragOver,
}: ChatComposerProps) {
  const { t } = useI18n()
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // Reset the auto-growing textarea height after a send empties the input.
  useEffect(() => {
    if (!input && textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }, [input])

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  function handleInput(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'
  }

  function handleFilePaste(e: React.ClipboardEvent) {
    const items = e.clipboardData?.files
    if (items && items.length > 0) {
      e.preventDefault()
      addFiles(items)
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files)
    }
  }

  function openFilePicker() {
    fileInputRef.current?.click()
  }

  function getFileTypeBadge(file: File): string {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    if (ext === '.docx') return 'DOCX'
    if (ext === '.xml') return 'XML'
    return 'PDF'
  }

  function formatFileSize(bytes: number): string {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  return (
    <>
      {selectedFiles.length > 0 && (
        <div className="file-chips-bar">
          {selectedFiles.map((file, i) => (
            <div key={`${file.name}-${i}`} className="file-chip">
              <span className={`file-chip-badge ${getFileTypeBadge(file).toLowerCase()}`}>
                {getFileTypeBadge(file)}
              </span>
              <span className="file-chip-name">{file.name}</span>
              <span className="file-chip-size">{formatFileSize(file.size)}</span>
              <button
                className="file-chip-remove"
                onClick={() => removeFile(i)}
                aria-label="Remove file"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}
      <div
        className="chat-input-wrapper"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="file-input-hidden"
          accept=".pdf,.docx,.xml,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/xml,text/xml"
          multiple
          onChange={e => { if (e.target.files) addFiles(e.target.files); e.target.value = '' }}
        />
        <button
          className="file-upload-btn"
          onClick={openFilePicker}
          aria-label="Attach patent files"
          title={t('chat.attachFiles') || 'Attach patent specification files (PDF, DOCX, XML)'}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
          </svg>
        </button>
        <textarea
          ref={textareaRef}
          className="chat-input"
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          onPaste={handleFilePaste}
          placeholder={t('chat.placeholder')}
          rows={1}
        />
        {streaming ? (
          <button
            className="send-btn"
            onClick={abort}
            style={{ background: 'var(--color-text-secondary)' }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="6" width="12" height="12" />
            </svg>
          </button>
        ) : (
          <button
            className="send-btn"
            onClick={() => send()}
            disabled={!input.trim() && selectedFiles.length === 0}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        )}
      </div>
    </>
  )
}
