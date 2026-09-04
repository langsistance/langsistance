'use client'

import { useState } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { getSceneMode, setSceneMode, type SceneMode } from '@/lib/sceneStore'
import ChatComposer, { type ChatComposerProps } from './ChatComposer'
import SceneHint from './SceneHint'
import SceneBar from './SceneBar'
import SellerLandingSections from './SellerLandingSections'
import PatentOnboardingWizard from './PatentOnboardingWizard'

// Linear icons matching the app's existing stroke-2px style (feather/lucide
// family). cap1 adds a star inside the magnifier for the US element.
const ICONS = {
  search: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="10.5" cy="10.5" r="6.5" />
      <path d="m21 21-4.3-4.3" />
      <path d="M10.5 7.6l.85 2 2 .85-2 .85-.85 2-.85-2-2-.85 2-.85z" fill="currentColor" stroke="none" />
    </svg>
  ),
  chat: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  ),
  clock: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 3" />
    </svg>
  ),
  globe: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9" />
      <path d="M3 12h18" />
      <path d="M12 3a15 15 0 0 1 0 18 15 15 0 0 1 0-18" />
    </svg>
  ),
  download: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3v12" />
      <path d="m7 10 5 5 5-5" />
      <path d="M4 21h16" />
    </svg>
  ),
  gift: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 12v10H4V12" />
      <path d="M2 7h20v5H2z" />
      <path d="M12 22V7" />
      <path d="M12 7H7.5a2.5 2.5 0 0 1 0-5C11 2 12 7 12 7Z" />
      <path d="M12 7h4.5a2.5 2.5 0 0 0 0-5C13 2 12 7 12 7Z" />
    </svg>
  ),
}

const CAPABILITIES = [
  { cap: 'us-search', icon: ICONS.search, titleKey: 'chat.landing.cap1Title', descKey: 'chat.landing.cap1Desc', free: false },
  { cap: 'nl-search', icon: ICONS.chat, titleKey: 'chat.landing.cap2Title', descKey: 'chat.landing.cap2Desc', free: false },
  { cap: 'prosecution', icon: ICONS.clock, titleKey: 'chat.landing.cap3Title', descKey: 'chat.landing.cap3Desc', free: false },
  { cap: 'family', icon: ICONS.globe, titleKey: 'chat.landing.cap4Title', descKey: 'chat.landing.cap4Desc', free: false },
  { cap: 'download', icon: ICONS.download, titleKey: 'chat.landing.cap5Title', descKey: 'chat.landing.cap5Desc', free: false },
  { cap: 'free', icon: ICONS.gift, titleKey: 'chat.landing.cap6Title', descKey: 'chat.landing.cap6Desc', free: true },
] as const

export default function ChatLanding(composerProps: ChatComposerProps) {
  const { t } = useI18n()
  const [mode, setMode] = useState<SceneMode>(() => getSceneMode())

  const handleSceneChange = (next: SceneMode) => {
    setMode(next)
    setSceneMode(next)
  }

  return (
    <div className="chat-landing">
      <h2 className="chat-landing-slogan">{t('chat.landing.slogan')}</h2>
      <div className="chat-landing-scene">
        <SceneBar mode={mode} onChange={handleSceneChange} />
      </div>
      <div className="chat-landing-composer">
        <ChatComposer {...composerProps} />
      </div>
      {mode === 'seller' ? (
        <SellerLandingSections />
      ) : (
      <div className="chat-landing-grid">
        {CAPABILITIES.map((cap) => (
          <div key={cap.titleKey} className={`chat-landing-card${cap.free ? ' free' : ''}`} data-cap={cap.cap}>
            <div className="chat-landing-card-icon">{cap.icon}</div>
            <div className="chat-landing-card-body">
              <h3 className="chat-landing-card-title">{t(cap.titleKey)}</h3>
              <p className="chat-landing-card-desc">{t(cap.descKey)}</p>
            </div>
          </div>
        ))}
      </div>
      )}
      {mode !== 'seller' && (
        <div className="chat-landing-section">
          <SceneHint />
        </div>
      )}
      <PatentOnboardingWizard />
    </div>
  )
}
