'use client'

import type { SceneMode } from '@/lib/sceneStore'

interface SceneBarProps {
  mode: SceneMode
  onChange: (mode: SceneMode) => void
}

const TABS: { key: SceneMode; label: string }[] = [
  { key: 'pro', label: '专业工作台' },
  { key: 'seller', label: '卖家安全台' },
]

/**
 * Scene switcher (专业工作台 ⇄ 卖家安全台).
 *
 * 工作台页签: 底部品牌色指示条 + 激活项浅 teal 底, 未激活项静默灰字——
 * 与整体"台"概念一致, 视觉明显区别于分段小开关。纯 UI 状态, 不改变 URL;
 * 流式发送时读持久化 scene(lib/useChatStream -> lib/sceneStore)。
 */
export default function SceneBar({ mode, onChange }: SceneBarProps) {
  return (
    <div
      className="inline-flex items-center gap-1 border-b border-gray-200"
      role="tablist"
      aria-label="工作台场景"
    >
      {TABS.map((tab) => {
        const active = mode === tab.key
        return (
          <button
            key={tab.key}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(tab.key)}
            className={`-mb-px rounded-t-lg border-b-2 px-6 py-2.5 text-sm transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500/50 ${
              active
                ? 'border-teal-600 bg-teal-50/70 font-semibold text-teal-700'
                : 'border-transparent text-gray-500 hover:bg-gray-50 hover:text-gray-800'
            }`}
          >
            {tab.label}
          </button>
        )
      })}
    </div>
  )
}
