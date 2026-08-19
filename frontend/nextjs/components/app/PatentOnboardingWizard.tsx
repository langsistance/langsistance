'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { useAuth } from '@/contexts/AuthContext'

const STORAGE_KEY_PREFIX = 'copiioai_patent_onboarding_done'

interface TourStep {
  /** CSS selectors for the element(s) to highlight — union-bounded when multiple */
  targets: string[]
  /** Tooltip position relative to the highlighted area */
  placement: 'bottom' | 'top' | 'right' | 'left'
  /** Title */
  titleKey: string
  /** Body (i18n key or hardcoded) */
  bodyKey: string
}

// ── Define what elements to point at on the chat landing page ──

function getTourSteps(t: (key: string, params?: Record<string, string | number>) => string, lang: string): TourStep[] {
  return [
    {
      // 落地页六大能力卡片：自然语言检索
      targets: ['[data-cap="nl-search"]'],
      placement: 'bottom',
      titleKey: lang === 'zh' ? '💬 自然语言检索' : '💬 Natural Language Search',
      bodyKey: lang === 'zh'
        ? '用日常语言描述你的检索需求——按申请号、专利号、公开号、关键词或权利人，AI 自动理解并为你找到专利。'
        : 'Describe your search in everyday language — by application number, publication number, keyword, or assignee. AI understands and finds the patents for you.',
    },
    {
      // 落地页六大能力卡片：审查历史分析 + 跨国同族专利审查历史分析（两张卡片一起高亮）
      targets: ['[data-cap="prosecution"]', '[data-cap="family"]'],
      placement: 'top',
      titleKey: lang === 'zh' ? '🔬 审查历史与跨国同族分析' : '🔬 Prosecution & Family Analysis',
      bodyKey: lang === 'zh'
        ? '深入分析单件专利的审查历史（OA 答复、权利要求演变），也可跨国家对比同族专利的审查过程，洞察授权策略。'
        : 'Dive deep into a patent’s prosecution history (OA responses, claim evolution), or compare prosecution across its global family members.',
    },
    {
      targets: ['.chat-input-wrapper'],
      placement: 'top',
      titleKey: lang === 'zh' ? '⌨️ 在这里输入问题' : '⌨️ Type Your Question Here',
      bodyKey: lang === 'zh'
        ? '在输入框中描述你的专利分析需求，按 Enter 发送。支持上传 PDF/DOCX/XML 专利说明书文件。现在就试试吧！'
        : 'Describe your patent analysis needs in the input box, press Enter to send. You can also upload PDF/DOCX/XML patent specification files. Try it now!',
    },
  ]
}

export default function PatentOnboardingWizard() {
  const { t, lang } = useI18n()
  const { user } = useAuth()
  const [visible, setVisible] = useState(false)
  const [ready, setReady] = useState(false)
  const [step, setStep] = useState(0)
  const [spotlight, setSpotlight] = useState({ x: 0, y: 0, w: 0, h: 0 })
  const [tooltipStyle, setTooltipStyle] = useState<Record<string, string>>({})

  const storageKey = `${STORAGE_KEY_PREFIX}_${user?.uid || 'unknown'}`
  const tourSteps = useMemo(() => getTourSteps(t, lang), [t, lang])

  // Check localStorage on mount
  useEffect(() => {
    try {
      if (localStorage.getItem(storageKey) === '1') return
    } catch {}
    setVisible(true)
  }, [storageKey])

  // Calculate spotlight + tooltip position for current step
  const updatePositions = useCallback(() => {
    const els = tourSteps[step].targets
      .map((s) => document.querySelector(s))
      .filter((el): el is Element => Boolean(el))
    if (els.length !== tourSteps[step].targets.length) {
      setReady(false)
      return
    }

    // Union bounding box across all targets (e.g. two capability cards)
    const union = els.reduce(
      (acc, el) => {
        const r = el.getBoundingClientRect()
        return {
          left: Math.min(acc.left, r.left),
          top: Math.min(acc.top, r.top),
          right: Math.max(acc.right, r.right),
          bottom: Math.max(acc.bottom, r.bottom),
        }
      },
      { left: Infinity, top: Infinity, right: -Infinity, bottom: -Infinity },
    )
    const rect = { ...union, width: union.right - union.left, height: union.bottom - union.top }
    const padding = 8
    setSpotlight({
      x: rect.left - padding,
      y: rect.top - padding,
      w: rect.width + padding * 2,
      h: rect.height + padding * 2,
    })

    // Tooltip position
    const placement = tourSteps[step].placement
    const gap = 16
    const ttW = 340

    let ttLeft = rect.left + rect.width / 2 - ttW / 2
    ttLeft = Math.max(12, Math.min(ttLeft, window.innerWidth - ttW - 12))

    const arrowOffset = rect.left + rect.width / 2 - ttLeft

    let style: Record<string, string> = {
      position: 'fixed',
      left: `${ttLeft}px`,
      maxWidth: `${ttW}px`,
      '--arrow-offset': `${arrowOffset}px`,
    }

    if (placement === 'bottom') {
      style.top = `${rect.bottom + gap}px`
    } else if (placement === 'top') {
      style.bottom = `${window.innerHeight - rect.top + gap}px`
    } else if (placement === 'right') {
      style.top = `${Math.max(12, rect.top + rect.height / 2 - 60)}px`
      style.left = `${rect.right + gap}px`
    } else {
      style.top = `${Math.max(12, rect.top + rect.height / 2 - 60)}px`
      style.left = `${rect.left - ttW - gap}px`
    }

    setTooltipStyle(style)
    setReady(true)
  }, [step, tourSteps])

  useEffect(() => {
    window.addEventListener('resize', updatePositions)
    window.addEventListener('scroll', updatePositions)
    return () => {
      window.removeEventListener('resize', updatePositions)
      window.removeEventListener('scroll', updatePositions)
    }
  }, [updatePositions])

  // When visible or step changes: poll for target element, then show
  useEffect(() => {
    if (!visible || step >= tourSteps.length) return

    setReady(false)
    const selectors = tourSteps[step].targets
    const timeouts: ReturnType<typeof setTimeout>[] = []
    let attempts = 0
    const maxAttempts = 30 // 3 seconds

    function tryShow() {
      const targets = selectors.map((s) => document.querySelector(s))
      if (targets.every(Boolean)) {
        // Scroll first + last so the whole union (e.g. two cards) is visible
        const first = targets[0] as Element
        const last = targets[targets.length - 1] as Element
        first.scrollIntoView({ behavior: 'smooth', block: 'center' })
        if (last !== first) last.scrollIntoView({ behavior: 'smooth', block: 'center' })
        updatePositions()
        return
      }
      attempts++
      if (attempts < maxAttempts) {
        timeouts.push(setTimeout(tryShow, 100))
      }
    }

    // Small initial delay in case React is still rendering
    timeouts.push(setTimeout(tryShow, 50))
    return () => timeouts.forEach(clearTimeout)
  }, [visible, step, tourSteps, updatePositions])

  if (!visible || !ready) return null

  function markDone() {
    setVisible(false)
    try { localStorage.setItem(storageKey, '1') } catch {}
  }

  function nextStep() {
    if (step >= tourSteps.length - 1) {
      markDone()
    } else {
      setStep(s => s + 1)
    }
  }

  function prevStep() {
    setStep(s => Math.max(0, s - 1))
  }

  const cur = tourSteps[step]

  return (
    <>
      {/* Spotlight cutout overlay */}
      <svg className="patent-tour-overlay" viewBox={`0 0 ${window.innerWidth} ${window.innerHeight}`}>
        <defs>
          <mask id="spotlight-mask">
            <rect width="100%" height="100%" fill="white" />
            <rect
              x={spotlight.x}
              y={spotlight.y}
              width={spotlight.w}
              height={spotlight.h}
              rx="12"
              fill="black"
            />
          </mask>
        </defs>
        <rect
          width="100%"
          height="100%"
          fill="rgba(0,0,0,0.5)"
          mask="url(#spotlight-mask)"
        />
        {/* Highlight border */}
        <rect
          x={spotlight.x}
          y={spotlight.y}
          width={spotlight.w}
          height={spotlight.h}
          rx="12"
          fill="none"
          stroke="rgba(16,163,127,0.6)"
          strokeWidth="2"
        />
      </svg>

      {/* Tooltip */}
      <div className="patent-tour-tooltip" style={tooltipStyle}>
        {/* Arrow */}
        <div className={`patent-tour-arrow patent-tour-arrow-${cur.placement}`} />

        {/* Step dots */}
        <div className="patent-tour-step-dots">
          {tourSteps.map((_, i) => (
            <span key={i} className={`patent-tour-dot${i === step ? ' active' : ''}${i < step ? ' done' : ''}`} />
          ))}
        </div>

        {/* Content */}
        <h3 className="patent-tour-tt-title">{cur.titleKey}</h3>
        <p className="patent-tour-tt-body">{cur.bodyKey}</p>

        {/* Actions */}
        <div className="patent-tour-actions">
          <button className="patent-tour-btn-skip" onClick={markDone}>
            {t('patentOnboarding.skip')}
          </button>
          <div style={{ display: 'flex', gap: 8 }}>
            {step > 0 && (
              <button className="patent-tour-btn-prev" onClick={prevStep}>
                {t('patentOnboarding.prev')}
              </button>
            )}
            <button className="patent-tour-btn-next" onClick={nextStep}>
              {step === tourSteps.length - 1 ? t('patentOnboarding.startChat') : t('patentOnboarding.next')}
            </button>
          </div>
        </div>
      </div>
    </>
  )
}
