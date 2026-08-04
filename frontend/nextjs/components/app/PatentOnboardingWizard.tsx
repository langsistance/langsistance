'use client'

import { useState, useEffect, useCallback } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { useAuth } from '@/contexts/AuthContext'

const STORAGE_KEY_PREFIX = 'copiioai_patent_onboarding_done'

interface TourStep {
  /** CSS selector for the element to highlight */
  target: string
  /** Tooltip position relative to target */
  placement: 'bottom' | 'top' | 'right' | 'left'
  /** Title */
  titleKey: string
  /** Body (i18n key or hardcoded) */
  bodyKey: string
}

// ── Define what elements to point at on the chat page ──

function getTourSteps(t: (key: string, params?: Record<string, string | number>) => string, lang: string): TourStep[] {
  return [
    {
      target: '.scene-hint-group-smart .knowledge-group-item',
      placement: 'bottom',
      titleKey: lang === 'zh' ? '💬 智能问答 — 示例提问' : '💬 Smart Q&A — Example',
      bodyKey: lang === 'zh'
        ? '这是智能问答的一个示例。你可以按申请号、专利号、公开号、关键词或权利人检索专利文档，点击即可快速开始。'
        : 'This is a Smart Q&A example. You can search patents by application number, patent number, publication number, keyword, or assignee. Click to try it.',
    },
    {
      target: '.scene-hint-group-deep .knowledge-group-item',
      placement: 'top',
      titleKey: lang === 'zh' ? '🔬 深度研究 — 示例提问' : '🔬 Deep Research — Example',
      bodyKey: lang === 'zh'
        ? '这是深度研究的一个示例。支持全球专利家族审查分析（跨国分析、OA答复、权利要求演变等）以及专利LLM驱动的批量分析。'
        : 'This is a Deep Research example. It covers global patent family prosecution analysis (cross-country, OA response, claim evolution, etc.) and patent LLM-powered batch analysis.',
    },
    {
      target: '.chat-input-wrapper',
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
  const tourSteps = getTourSteps(t, lang)

  // Check localStorage on mount
  useEffect(() => {
    try {
      if (localStorage.getItem(storageKey) === '1') return
    } catch {}
    setVisible(true)
  }, [storageKey])

  // Calculate spotlight + tooltip position for current step
  const updatePositions = useCallback(() => {
    const target = document.querySelector(tourSteps[step].target)
    if (!target) {
      setReady(false)
      return
    }

    const rect = target.getBoundingClientRect()
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
    const selector = tourSteps[step].target
    let attempts = 0
    const maxAttempts = 30 // 3 seconds

    function tryShow() {
      const target = document.querySelector(selector)
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'center' })
        updatePositions()
        return
      }
      attempts++
      if (attempts < maxAttempts) {
        setTimeout(tryShow, 100)
      }
    }

    // Small initial delay in case React is still rendering
    setTimeout(tryShow, 50)
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
