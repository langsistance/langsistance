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
 * 两页签常显细框: 激活 = teal 框 + 浅 teal 底 + 品牌字; 未激活 = 浅灰框 + 灰字,
 * hover 灰框加深/底色微现。无下划线、无整段灰容器, 平铺双页签。纯 UI 状态,
 * 不改变 URL; 流式发送时读持久化 scene(sceneStore)。
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
                ? 'border-teal-300 bg-teal-50 font-medium text-teal-700'
                : 'border-gray-200 text-gray-600 hover:border-gray-300 hover:bg-gray-50 hover:text-gray-800'
            }`}
          >
            {tab.label}
          </button>
        )
      })}
    </div>
  )
}
