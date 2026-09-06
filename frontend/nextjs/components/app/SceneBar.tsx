'use client'

import type { ReactNode } from 'react'
import type { SceneMode } from '@/lib/sceneStore'

interface SceneBarProps {
  mode: SceneMode
  onChange: (mode: SceneMode) => void
}

function ProIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="2" y="7" width="20" height="14" rx="2" />
      <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
    </svg>
  )
}

function SellerIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 22s8-3.6 8-10V5.2L12 2 4 5.2V12c0 6.4 8 10 8 10z" />
      <path d="m9 11.6 2.2 2.2L15.4 9.6" />
    </svg>
  )
}

const TABS: { key: SceneMode; label: string; icon: () => ReactNode }[] = [
  { key: 'pro', label: '专业工作台', icon: ProIcon },
  { key: 'seller', label: '卖家安全台', icon: SellerIcon },
]

/**
 * Scene switcher (专业工作台 ⇄ 卖家安全台).
 *
 * DeepSeek instant-expert / WorkBuddy 式工作台页签: 图标 + 标签 + 底部品牌色
 * 指示条, 激活项浅 teal 底——图标与条状激活态共同给出"可切换页签"暗示。
 * 纯 UI 状态, 不改变 URL; 流式发送时读持久化 scene(sceneStore)。
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
        const Icon = tab.icon
        return (
          <button
            key={tab.key}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(tab.key)}
            className={`-mb-px flex cursor-pointer items-center gap-2 rounded-t-lg border-b-2 px-5 py-2.5 text-sm transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500/50 ${
              active
                ? 'border-teal-600 bg-teal-50/70 font-semibold text-teal-700'
                : 'border-transparent text-gray-500 hover:bg-gray-50 hover:text-gray-800'
            }`}
          >
            <Icon />
            {tab.label}
          </button>
        )
      })}
    </div>
  )
}
