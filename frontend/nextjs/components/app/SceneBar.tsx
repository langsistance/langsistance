'use client'

import type { SceneMode } from '@/lib/sceneStore'

interface SceneBarProps {
  mode: SceneMode
  onChange: (mode: SceneMode) => void
}

/**
 * Scene switcher (专业工作台 ⇄ 卖家安全台).
 *
 * Default scene is 专业工作台 (spec §3.2 / 2026-09-04 反馈); switching is
 * pure UI state — the URL never changes and the streaming hook reads the
 * persisted mode at send time (lib/useChatStream -> lib/sceneStore).
 */
export default function SceneBar({ mode, onChange }: SceneBarProps) {
  const buttonClass = (active: boolean) =>
    `px-4 py-2 rounded-md text-sm font-medium transition ${
      active
        ? 'bg-teal-600 text-white shadow-sm'
        : 'text-gray-600 hover:text-gray-900'
    }`

  return (
    <div
      className="inline-flex rounded-lg bg-gray-100 p-1"
      role="tablist"
      aria-label="工作台场景"
    >
      <button
        type="button"
        role="tab"
        aria-selected={mode === 'pro'}
        onClick={() => onChange('pro')}
        className={buttonClass(mode === 'pro')}
      >
        专业工作台
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={mode === 'seller'}
        onClick={() => onChange('seller')}
        className={buttonClass(mode === 'seller')}
      >
        卖家安全台
      </button>
    </div>
  )
}
