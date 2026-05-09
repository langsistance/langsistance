'use client'

import { useState, useEffect } from 'react'
import { onAuthStateChanged } from 'firebase/auth'
import type { User } from 'firebase/auth'
import { auth } from '@/lib/firebase'
import { useLandingI18n, LANGUAGE_NAMES, type LangKey } from '@/lib/landing-i18n'

export default function LandingHeader() {
  const { lang, setLanguage, t } = useLandingI18n()
  const [user, setUser] = useState<User | null | undefined>(undefined)

  useEffect(() => {
    return onAuthStateChanged(auth, setUser)
  }, [])

  return (
    <header className="fixed top-0 left-0 right-0 bg-white shadow-sm z-50">
      <nav className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <img src="/logo.png" alt="CopiioAI Logo" className="w-8 h-8 rounded-full" />
          <span className="text-xl font-semibold text-gray-900">CopiioAI</span>
        </div>

        <div className="flex items-center space-x-4">
          <a
            href="https://chromewebstore.google.com/detail/copiioai/lejbegpfaanpcilacmakkdediinkmnne"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center space-x-2 px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
            </svg>
            <span>{t('header.extension')}</span>
          </a>

          <button
            disabled
            className="flex items-center space-x-2 px-4 py-2 text-gray-400 border border-gray-200 rounded-lg cursor-not-allowed"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
            <span>{t('header.desktop')}</span>
          </button>

          <a
            href="https://copiioaicom-spec.github.io/CopiioAI-Natural-language-interface-for-accessing-internet-data/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center space-x-2 px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
            <span>Documentation</span>
          </a>

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
              Sign In
            </a>
          ) : (
            <a
              href="/app/chat"
              className="px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition"
            >
              My Workspace
            </a>
          )}
        </div>
      </nav>
    </header>
  )
}
