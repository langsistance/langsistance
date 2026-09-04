'use client'

import { useState } from 'react'
import { getSceneMode, setSceneMode, type SceneMode } from '@/lib/sceneStore'

interface SceneBarProps {
  /** Called with a preset seller question when a quick chip is clicked. */
  onQuickQuestion?: (text: string) => void
}

const SELLER_CHIPS = ['折叠水杯', '遥控玩具蛇', '便携榨汁杯能上美国亚马逊吗']

/**
 * Scene switcher (卖家安全台 ⇄ 专业工作台) + seller quick chips.
 *
 * Scene is pure UI state — switching never changes the URL (spec §3.2-1);
 * the streaming hook reads the persisted mode at send time
 * (lib/useChatStream -> lib/sceneStore).
 */
export default function SceneBar({ onQuickQuestion }: SceneBarProps) {
  const [mode, setMode] = useState<SceneMode>(() => getSceneMode())

  const switchMode = (next: SceneMode) => {
    setMode(next)
    setSceneMode(next)
  }

  const buttonClass = (active: boolean) =>
    `px-4 py-2 rounded-md text-sm font-medium transition ${
      active
        ? 'bg-teal-600 text-white shadow-sm'
        : 'text-gray-600 hover:text-gray-900'
    }`

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="inline-flex rounded-lg bg-gray-100 p-1" role="tablist" aria-label="工作台场景">
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'seller'}
          onClick={() => switchMode('seller')}
          className={buttonClass(mode === 'seller')}
        >
          卖家安全台
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === 'pro'}
          onClick={() => switchMode('pro')}
          className={buttonClass(mode === 'pro')}
        >
          专业工作台
        </button>
      </div>

      {mode === 'seller' && (
        <div className="flex flex-wrap justify-center gap-2">
          {SELLER_CHIPS.map((chip) => (
            <button
              key={chip}
              type="button"
              onClick={() => onQuickQuestion?.(chip)}
              className="rounded-full border border-gray-300 px-3.5 py-1.5 text-xs text-gray-600 hover:border-teal-500 hover:text-teal-600 transition"
            >
              {chip}
            </button>
          ))}
          <span className="self-center text-[11px] text-gray-400">
            卖家模式已开启：结论用人话讲，不出现专利行话
          </span>
        </div>
      )}
    </div>
  )
}
