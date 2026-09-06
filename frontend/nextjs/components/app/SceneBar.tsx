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
 * 深浅/边框区分: 激活页签 = 细 teal 边框 + 浅 teal 底; 未激活 = 无边框无底,
 * hover 微灰。无下划线、无整段灰容器, 平铺双页签。纯 UI 状态, 不改变 URL;
 * 流式发送时读持久化 scene(sceneStore)。
 */
export default function SceneBar({ mode, onChange }: SceneBarProps) {
  return (
    <div
      className="inline-flex items-center gap-1.5"
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
            className={`cursor-pointer rounded-xl border px-5 py-2 text-sm transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500/50 ${
              active
                ? 'border-teal-200 bg-teal-50 font-medium text-teal-700'
                : 'border-transparent text-gray-500 hover:bg-gray-100/70 hover:text-gray-800'
            }`}
          >
            {tab.label}
          </button>
        )
      })}
    </div>
  )
}
