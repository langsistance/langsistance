'use client'

import { useState, useEffect } from 'react'
import { useI18n } from '@/lib/app-i18n'

type Capability = 'smartQA' | 'deepResearch' | null

const STORAGE_KEY = 'copiioai_patent_onboarding_done'

function PatentOnboardingWizard() {
  const { t, lang } = useI18n()
  const [visible, setVisible] = useState(false)
  const [step, setStep] = useState(1)
  const [selectedCap, setSelectedCap] = useState<Capability>(null)
  const [prefillText, setPrefillText] = useState('')

  // Check localStorage on mount
  useEffect(() => {
    try {
      if (localStorage.getItem(STORAGE_KEY) === '1') return
    } catch {}
    setVisible(true)
  }, [])

  // ── handlers (plain functions, no useCallback) ──

  function markDone() {
    setVisible(false)
    try { localStorage.setItem(STORAGE_KEY, '1') } catch {}
  }

  function handleExampleClick(example: string) {
    setPrefillText(example)
    setStep(3)
  }

  function handleSend() {
    const text = prefillText.trim()
    if (!text) return
    markDone()
    const textarea = document.querySelector('.chat-input') as HTMLTextAreaElement
    if (textarea) {
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value'
      )?.set
      if (setter) {
        setter.call(textarea, text)
      } else {
        textarea.value = text
      }
      textarea.dispatchEvent(new Event('input', { bubbles: true }))
      setTimeout(() => {
        const sendBtn = document.querySelector('.send-btn') as HTMLButtonElement
        if (sendBtn && !sendBtn.disabled) sendBtn.click()
      }, 100)
    }
  }

  // ── early return AFTER all hooks ──

  if (!visible) return null

  // ── data ──

  const smartQAItems = [
    { label: lang === 'zh' ? '申请号' : 'Application No.', example: 'Retrieve all patent documents for application number 18893954.' },
    { label: lang === 'zh' ? '专利号' : 'Patent No.', example: 'Search using patent number 12,615,916.' },
    { label: lang === 'zh' ? '公开号' : 'Publication No.', example: 'Search using publication number US20240121982A1.' },
    { label: lang === 'zh' ? '关键词' : 'Keyword', example: 'Search the USPTO for patents using the keyword "Agentic AI".' },
    { label: lang === 'zh' ? '权利人' : 'Assignor', example: 'Search for patents where the assignee is Apple Inc.' },
    { label: lang === 'zh' ? 'Google搜索' : 'Google Search', example: 'Search Google for patents related to AI agents.' },
    { label: lang === 'zh' ? '文档检索' : 'Doc Retrieval', example: 'Retrieve all patent documents with publication number US20250103146A1.' },
  ]

  const deepResearchFamily = [
    { label: lang === 'zh' ? '跨国分析' : 'Cross-country', example: 'Analyze prosecution differences between US12506212 and its global family members' },
    { label: lang === 'zh' ? 'OA答复分析' : 'OA Response', example: 'Analyze rejection reasons and applicant response strategies for US12506212' },
    { label: lang === 'zh' ? '权利要求演变' : 'Claim Evolution', example: 'Analyze claim amendments and scope changes of US12506212 and its family members' },
    { label: lang === 'zh' ? '授权策略' : 'Grant Strategy', example: 'Analyze why US12506212 was granted and key amendment strategies' },
    { label: lang === 'zh' ? '无效风险' : 'Invalidity & Risk', example: 'Find limiting statements and legal risks in the prosecution history of US12506212' },
  ]

  const deepResearchBatch = [
    { label: lang === 'zh' ? '搜索分析' : 'Search Analysis', example: 'What recent patents does Tesla have in autonomous driving' },
    { label: lang === 'zh' ? '指定专利分析' : 'Specified Patent', example: 'Analyze patents 17429113, 18012525, 18331482' },
    { label: lang === 'zh' ? '文件上传分析' : 'File Upload', example: 'Upload specification files (PDF/XML/DOCX) for text analysis. Filter these documents for AI-related patents' },
    { label: lang === 'zh' ? '追问分析' : 'Follow-up', example: 'Query and filter previously retrieved patent results. Which of these are AI-related patents' },
  ]

  const steps = [1, 2, 3]

  // ── render ──

  return (
    <div className="patent-onboard-overlay">
      <div className="patent-onboard-wizard">
        {/* Header */}
        <div className="patent-onboard-header">
          <div className="patent-onboard-step-indicator">
            <span className="patent-onboard-step-label">
              {t('patentOnboarding.step', { current: step })} {t('patentOnboarding.of', { total: 3 })}
            </span>
            {steps.map((s, i) => (
              <span key={s}>
                {i > 0 && (
                  <span className={`patent-onboard-step-line${s <= step ? ' done' : ''}`} />
                )}
                <span className={`patent-onboard-step-dot${s < step ? ' done' : ''}${s === step ? ' active' : ''}`}>
                  {s < step ? '✓' : s}
                </span>
              </span>
            ))}
          </div>
          <button className="patent-onboard-skip-btn" onClick={markDone}>
            {t('patentOnboarding.skip')}
          </button>
        </div>

        {/* Body */}
        <div className="patent-onboard-body">
          {step === 1 && (
            <>
              <h2 className="patent-onboard-title">{t('patentOnboarding.step1.title')}</h2>
              <p className="patent-onboard-subtitle">{t('patentOnboarding.step1.subtitle')}</p>
              <div className="patent-onboard-cards">
                <button
                  className={`patent-onboard-card${selectedCap === 'smartQA' ? ' selected' : ''}`}
                  onClick={() => setSelectedCap('smartQA')}
                >
                  <span className="patent-onboard-card-icon">💬</span>
                  <span className="patent-onboard-card-name">{t('patentOnboarding.step1.smartQA.name')}</span>
                  <span className="patent-onboard-card-desc">{t('patentOnboarding.step1.smartQA.desc')}</span>
                </button>
                <button
                  className={`patent-onboard-card${selectedCap === 'deepResearch' ? ' selected' : ''}`}
                  onClick={() => setSelectedCap('deepResearch')}
                >
                  <span className="patent-onboard-card-icon">🔬</span>
                  <span className="patent-onboard-card-name">{t('patentOnboarding.step1.deepResearch.name')}</span>
                  <span className="patent-onboard-card-desc">{t('patentOnboarding.step1.deepResearch.desc')}</span>
                </button>
              </div>
            </>
          )}

          {step === 2 && (
            <>
              <h2 className="patent-onboard-title">{t('patentOnboarding.step2.title')}</h2>
              <p className="patent-onboard-subtitle">{t('patentOnboarding.step2.tip')}</p>

              {selectedCap === 'smartQA' && (
                <div className="patent-onboard-examples">
                  {smartQAItems.map((item, i) => (
                    <button key={i} className="patent-onboard-example-item" onClick={() => handleExampleClick(item.example)}>
                      <span className="patent-onboard-example-label">{item.label}</span>
                      <span className="patent-onboard-example-text">{item.example}</span>
                    </button>
                  ))}
                </div>
              )}

              {selectedCap === 'deepResearch' && (
                <div className="patent-onboard-examples">
                  <div className="patent-onboard-example-group">
                    <h3 className="patent-onboard-group-title">{t('patentOnboarding.step2.familyAnalysis')}</h3>
                    {deepResearchFamily.map((item, i) => (
                      <button key={i} className="patent-onboard-example-item" onClick={() => handleExampleClick(item.example)}>
                        <span className="patent-onboard-example-label">{item.label}</span>
                        <span className="patent-onboard-example-text">{item.example}</span>
                      </button>
                    ))}
                  </div>
                  <div className="patent-onboard-example-group">
                    <h3 className="patent-onboard-group-title">{t('patentOnboarding.step2.batchAnalysis')}</h3>
                    {deepResearchBatch.map((item, i) => (
                      <button key={i} className="patent-onboard-example-item" onClick={() => handleExampleClick(item.example)}>
                        <span className="patent-onboard-example-label">{item.label}</span>
                        <span className="patent-onboard-example-text">{item.example}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}

          {step === 3 && (
            <>
              <h2 className="patent-onboard-title">{t('patentOnboarding.step3.title')}</h2>
              <p className="patent-onboard-subtitle">{t('patentOnboarding.step3.subtitle')}</p>
              <div className="patent-onboard-input-area">
                <div className="patent-onboard-input-wrapper">
                  <textarea
                    className="patent-onboard-textarea"
                    placeholder={t('patentOnboarding.step3.placeholder')}
                    value={prefillText}
                    onChange={e => setPrefillText(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        handleSend()
                      }
                    }}
                    rows={3}
                  />
                  <button className="patent-onboard-send-btn" onClick={handleSend} disabled={!prefillText.trim()} title="Send">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <line x1="22" y1="2" x2="11" y2="13" />
                      <polygon points="22 2 15 22 11 13 2 9 22 2" />
                    </svg>
                  </button>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="patent-onboard-footer">
          {step > 1 ? (
            <button className="patent-onboard-btn-prev" onClick={() => setStep(step - 1)}>
              ← {t('patentOnboarding.prev')}
            </button>
          ) : (
            <div />
          )}
          {step === 1 && (
            <button className="patent-onboard-btn-next" disabled={!selectedCap} onClick={() => setStep(2)}>
              {t('patentOnboarding.next')}
            </button>
          )}
          {step === 2 && (
            <button className="patent-onboard-btn-next" onClick={() => setStep(3)}>
              {t('patentOnboarding.next')}
            </button>
          )}
          {step === 3 && (
            <button className="patent-onboard-btn-next" onClick={markDone}>
              {t('patentOnboarding.startChat')}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export default PatentOnboardingWizard
