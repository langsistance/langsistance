'use client'

import { useI18n } from '@/lib/app-i18n'
import ChatComposer, { type ChatComposerProps } from './ChatComposer'
import SceneHint from './SceneHint'

const CAPABILITIES = [
  { titleKey: 'chat.landing.cap1Title', descKey: 'chat.landing.cap1Desc', free: false },
  { titleKey: 'chat.landing.cap2Title', descKey: 'chat.landing.cap2Desc', free: false },
  { titleKey: 'chat.landing.cap3Title', descKey: 'chat.landing.cap3Desc', free: false },
  { titleKey: 'chat.landing.cap4Title', descKey: 'chat.landing.cap4Desc', free: false },
  { titleKey: 'chat.landing.cap5Title', descKey: 'chat.landing.cap5Desc', free: false },
  { titleKey: 'chat.landing.cap6Title', descKey: 'chat.landing.cap6Desc', free: true },
] as const

export default function ChatLanding(composerProps: ChatComposerProps) {
  const { t } = useI18n()

  return (
    <div className="chat-landing">
      <h2 className="chat-landing-slogan">{t('chat.landing.slogan')}</h2>
      <div className="chat-landing-composer">
        <ChatComposer {...composerProps} />
      </div>
      <div className="chat-landing-grid">
        {CAPABILITIES.map((cap) => (
          <div key={cap.titleKey} className={`chat-landing-card${cap.free ? ' free' : ''}`}>
            <h3 className="chat-landing-card-title">{t(cap.titleKey)}</h3>
            <p className="chat-landing-card-desc">{t(cap.descKey)}</p>
          </div>
        ))}
      </div>
      <div className="chat-landing-section">
        <h3 className="chat-landing-section-title">{t('chat.landing.sectionTitle')}</h3>
        <SceneHint />
      </div>
    </div>
  )
}
