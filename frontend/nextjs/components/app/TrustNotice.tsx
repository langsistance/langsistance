'use client'

import { useI18n } from '@/lib/app-i18n'
import { TRUST_COPY, TRUST_NOTES } from '@/lib/trustNotice'

// 线性图标，风格对齐 ChatLanding.tsx 的 ICONS（stroke 2px，24 视图框）
const ICONS: Record<string, JSX.Element> = {
  train: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="11" width="18" height="11" rx="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  ),
  isolate: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  ),
  upload: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 6h18" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </svg>
  ),
}

// 用字面量联合而非从 .js 模块 import 类型：trustNotice.js 是纯 JS，
// 类型靠 JSDoc，跨文件 typeof import 容易在 next build 下解析失败。
interface TrustNoticeProps {
  variant?: 'card' | 'inline' | 'hint'
}

export default function TrustNotice({ variant = 'card' }: TrustNoticeProps) {
  const { lang } = useI18n()
  const locale = lang === 'en' ? 'en' : 'zh'

  if (variant === 'inline') {
    return <p className="trust-inline">{TRUST_NOTES[locale].login}</p>
  }

  if (variant === 'hint') {
    return <p className="trust-hint">{TRUST_NOTES[locale].focus}</p>
  }

  const copy = TRUST_COPY[locale]

  return (
    <section className="trust-notice" aria-label={copy.headline}>
      <h3 className="trust-notice-headline">{copy.headline}</h3>
      <ul className="trust-notice-items">
        {copy.items.map((item) => (
          <li key={item.key} className="trust-notice-item">
            <span className="trust-notice-icon">{ICONS[item.key]}</span>
            <span className="trust-notice-label">{item.label}</span>
            <span className="trust-notice-desc">{item.desc}</span>
          </li>
        ))}
      </ul>
      <a className="trust-notice-more" href="/security">{copy.more} →</a>
    </section>
  )
}
