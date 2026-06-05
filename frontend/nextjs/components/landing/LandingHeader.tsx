'use client'

import { useState, useEffect, useRef } from 'react'
import { onAuthChange, type AuthUser } from '@/lib/auth-client'
import { useLandingI18n, LANGUAGE_NAMES, type LangKey } from '@/lib/landing-i18n'
import { LANDING_HEADER_ACTIONS } from '@/lib/landingExamples'

export default function LandingHeader() {
  const { lang, setLanguage, t } = useLandingI18n()
  const [user, setUser] = useState<AuthUser | null | undefined>(undefined)
  const [examplesOpen, setExamplesOpen] = useState(false)
  const examplesMenuRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    return onAuthChange(setUser)
  }, [])

  useEffect(() => {
    if (!examplesOpen) {
      return undefined
    }

    function handlePointerDown(event: PointerEvent) {
      const target = event.target
      if (target instanceof Node && !examplesMenuRef.current?.contains(target)) {
        setExamplesOpen(false)
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setExamplesOpen(false)
      }
    }

    document.addEventListener('pointerdown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)

    return () => {
      document.removeEventListener('pointerdown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [examplesOpen])

  return (
    <header className="fixed top-0 left-0 right-0 bg-white shadow-sm z-50">
      <nav className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <img src="/logo.png" alt="CopiioAI Logo" className="w-8 h-8 rounded-full" />
          <span className="text-xl font-semibold text-gray-900">CopiioAI</span>
        </div>

        <div className="flex items-center space-x-4">
          {LANDING_HEADER_ACTIONS.map((action) => {
            const actionLabel = t(action.labelKey) || action.fallbackLabel

            if (action.type === 'link') {
              return (
                <a
                  key={action.labelKey}
                  href={action.href}
                  target={action.external ? '_blank' : undefined}
                  rel={action.external ? 'noopener noreferrer' : undefined}
                  className="flex items-center space-x-2 px-4 py-2 bg-white text-gray-700 hover:text-gray-900 border border-gray-300 rounded-lg hover:bg-gray-50 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                  <span>{actionLabel}</span>
                </a>
              )
            }

            return (
              <div key={action.labelKey} ref={examplesMenuRef} className="relative">
                <button
                  type="button"
                  onClick={() => setExamplesOpen((open) => !open)}
                  className="flex items-center space-x-2 px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2"
                  aria-controls="landing-examples-menu"
                  aria-expanded={examplesOpen}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M9 12h6m-6 4h6M8 4h8l4 4v12a2 2 0 01-2 2H8a2 2 0 01-2-2V6a2 2 0 012-2z" />
                  </svg>
                  <span>{actionLabel}</span>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {examplesOpen ? (
                  <div
                    id="landing-examples-menu"
                    className="absolute right-0 top-full mt-2 w-80 max-w-[calc(100vw-2rem)] bg-white border border-gray-200 rounded-lg shadow-lg py-2 z-10"
                  >
                    {(action.items || []).map((example) => (
                      <a
                        key={example.slug}
                        href={example.href}
                        className="block px-4 py-3 hover:bg-gray-50 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-teal-500"
                        onClick={() => setExamplesOpen(false)}
                      >
                        <span className="block text-sm font-semibold text-gray-900">
                          {example.titleKey ? t(example.titleKey) : example.title}
                        </span>
                        <span className="block text-xs text-gray-600 mt-1">
                          {example.summaryKey ? t(example.summaryKey) : example.summary}
                        </span>
                      </a>
                    ))}
                  </div>
                ) : null}
              </div>
            )
          })}

          {/* Language selector */}
          <div className="relative group">
            <button className="flex items-center space-x-2 px-4 py-2 text-gray-700 hover:text-gray-900 border border-gray-300 rounded-lg transition">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
              </svg>
              <span>{LANGUAGE_NAMES[lang]}</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            <div className="absolute right-0 top-full mt-1 hidden group-hover:block bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-10 min-w-[140px]">
              {(Object.keys(LANGUAGE_NAMES) as LangKey[]).map((code) => (
                <button
                  key={code}
                  onClick={() => setLanguage(code)}
                  className={`w-full text-left px-4 py-2 text-sm hover:bg-gray-50 ${lang === code ? 'font-semibold text-teal-600' : 'text-gray-700'}`}
                >
                  {LANGUAGE_NAMES[code]}
                </button>
              ))}
            </div>
          </div>

          {/* Auth-aware button */}
          {user === undefined ? null : user === null ? (
            <a
              href="/app/login"
              className="px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition"
            >
              {t('header.signin')}
            </a>
          ) : (
            <a
              href="/app/chat"
              className="px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition"
            >
              {t('header.workspace')}
            </a>
          )}
        </div>
      </nav>
    </header>
  )
}
