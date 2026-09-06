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
 * DeepSeek instant-expert / WorkBuddy 式页签: 灰底浅容器内, 激活项呈
 * 悬浮白卡 + 品牌色文字(teal), 未激活项静默灰字。纯 UI 状态, 不改变 URL;
 * 流式发送时读持久化 scene(lib/useChatStream -> lib/sceneStore)。
 */
export default function SceneBar({ mode, onChange }: SceneBarProps) {
  return (
    <div
      className="inline-flex items-center gap-1 rounded-2xl border border-gray-200/80 bg-gray-100/80 p-1.5"
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
            className={`rounded-xl px-5 py-2.5 text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500/50 ${
              active
                ? 'bg-white text-teal-700 shadow-sm ring-1 ring-gray-200'
                : 'text-gray-500 hover:bg-white/60 hover:text-gray-800'
            }`}
          >
            {tab.label}
          </button>
        )
      })}
    </div>
  )
}
