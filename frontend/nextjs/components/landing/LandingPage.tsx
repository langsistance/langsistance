'use client'

import { useState } from 'react'
import { useLandingI18n } from '@/lib/landing-i18n'
import LandingHeader from './LandingHeader'

function YouTubeFacade({ videoId, ariaLabel }: { videoId: string; ariaLabel: string }) {
  const [loaded, setLoaded] = useState(false)
  if (loaded) {
    return (
      <iframe
        className="w-full h-full"
        src={`https://www.youtube.com/embed/${videoId}?autoplay=1`}
        title={ariaLabel}
        allow="autoplay; encrypted-media"
        allowFullScreen
      />
    )
  }
  return (
    <div
      className="youtube-facade w-full h-full relative bg-black cursor-pointer flex items-center justify-center"
      style={{
        backgroundImage: `url('https://img.youtube.com/vi/${videoId}/maxresdefault.jpg')`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
      }}
    >
      <button
        onClick={() => setLoaded(true)}
        className="youtube-play-btn absolute inset-0 flex items-center justify-center"
        aria-label={ariaLabel}
      >
        <svg width="68" height="48" viewBox="0 0 68 48" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M66.52 7.74C65.73 4.79 63.4 2.46 60.45 1.67 55.19 0 34 0 34 0S12.82 0 7.55 1.67C4.6 2.46 2.27 4.79 1.48 7.74 0 13.01 0 24 0 24s0 10.99 1.48 16.26c.79 2.95 3.12 5.28 6.07 6.07C12.82 48 34 48 34 48s21.18 0 26.45-1.67c2.95-.79 5.28-3.12 6.07-6.07C68 34.99 68 24 68 24s0-10.99-1.48-16.26z"
            fill="#FF0000"
          />
          <path d="M27 34l18-10-18-10v20z" fill="white" />
        </svg>
      </button>
    </div>
  )
}

export default function LandingPage() {
  const { t } = useLandingI18n()

  return (
    <>
      <LandingHeader />
      <main className="pt-20">
        {/* Hero Section */}
        <section className="max-w-7xl mx-auto px-6 py-20 text-center">
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6">
            <span>{t('hero.title1')}</span><br />
            <span>{t('hero.title2')}</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto mb-12">
            <span>{t('hero.subtitle1')}</span><br />
            <span>{t('hero.subtitle2')}</span>
          </p>
          <div className="max-w-4xl mx-auto bg-white rounded-2xl shadow-2xl overflow-hidden">
            <div className="aspect-video">
              <YouTubeFacade videoId="WC3b299li7U" ariaLabel="Play CopiioAI Introduction Video" />
            </div>
          </div>
        </section>

        {/* What is CopiioAI Section */}
        <section className="bg-white py-20">
          <div className="max-w-7xl mx-auto px-6">
            <h2 className="text-4xl font-bold text-gray-900 text-center mb-16">{t('what.title')}</h2>
            <div className="grid md:grid-cols-2 gap-12 items-center">
              <div className="space-y-6">
                <p className="text-lg text-gray-700">{t('what.p1')}</p>
                <p className="text-lg text-gray-700">{t('what.p2')}</p>
                <p className="text-lg font-semibold text-gray-900">{t('what.p3')}</p>
              </div>
              <div className="rounded-2xl shadow-xl overflow-hidden">
                <img
                  src="https://images.unsplash.com/photo-1762340277380-04c2c30d0ef8?w=800&h=600&fit=crop&fm=webp"
                  alt="Conversational AI Interface showing chat-based API interaction"
                  className="w-full h-full object-cover aspect-video"
                  width={800}
                  height={600}
                />
              </div>
            </div>
          </div>
        </section>

        {/* How to Use CopiioAI Section */}
        <section className="py-20 bg-white">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-gray-900 mb-6">{t('howto.title')}</h2>
              <p className="text-xl text-gray-600 max-w-3xl mx-auto">{t('howto.subtitle')}</p>
            </div>
            <div className="grid md:grid-cols-2 gap-8 mb-12">
              {/* For Developers */}
              <div>
                <div className="text-center mb-4">
                  <span className="inline-block px-4 py-2 bg-blue-100 text-blue-800 rounded-full text-sm font-semibold">{t('dev.label')}</span>
                  <h3 className="text-2xl font-bold text-gray-900 mt-4">{t('dev.title')}</h3>
                </div>
                <div className="bg-white rounded-xl shadow-xl overflow-hidden">
                  <div className="aspect-video">
                    <YouTubeFacade videoId="mAIdNHtDIw8" ariaLabel="Play: How to Turn Any API into a Chat Interface with CopiioAI" />
                  </div>
                </div>
              </div>
              {/* For Non-Developers */}
              <div>
                <div className="text-center mb-4">
                  <span className="inline-block px-4 py-2 bg-teal-100 text-teal-800 rounded-full text-sm font-semibold">{t('nondev.label')}</span>
                  <h3 className="text-2xl font-bold text-gray-900 mt-4">{t('nondev.title')}</h3>
                </div>
                <div className="bg-white rounded-xl shadow-xl overflow-hidden">
                  <div className="aspect-video">
                    <YouTubeFacade videoId="XXoMGPtarsg" ariaLabel="Play: A Real-Time Data AI Assistant for Non-Developers" />
                  </div>
                </div>
              </div>
            </div>

            {/* Mid-page CTA Box */}
            <div className="bg-gradient-to-r from-teal-600 to-blue-600 rounded-2xl p-10 text-center">
              <h3 className="text-3xl font-bold text-white mb-4">{t('cta.main')}</h3>
              <p className="text-xl text-white opacity-90 mb-8 max-w-2xl mx-auto">
                <span>{t('cta.subtitle1')}</span><br />
                <span>{t('cta.subtitle2')}</span>
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                <a
                  href="https://chromewebstore.google.com/detail/copiioai/lejbegpfaanpcilacmakkdediinkmnne"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center space-x-2 px-8 py-4 bg-white text-teal-600 rounded-lg hover:bg-gray-100 transition font-semibold text-lg shadow-xl"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  <span>{t('cta.download')}</span>
                </a>
                <span className="text-white opacity-75">{t('cta.search')}</span>
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="py-20">
          <div className="max-w-7xl mx-auto px-6">
            <h2 className="text-4xl font-bold text-gray-900 text-center mb-16">{t('features.title')}</h2>
            <div className="grid md:grid-cols-2 gap-12 items-center">
              <div className="rounded-2xl overflow-hidden shadow-xl">
                <img
                  src="https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=800&h=600&fit=crop&fm=webp"
                  alt="Team Collaboration using CopiioAI chat-based API tools"
                  className="w-full h-full object-cover"
                  loading="lazy"
                  width={800}
                  height={600}
                />
              </div>
              <div className="space-y-8">
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">{t('features.f1.title')}</h3>
                  <p className="text-gray-600">{t('features.f1.desc')}</p>
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">{t('features.f2.title')}</h3>
                  <p className="text-gray-600">{t('features.f2.desc')}</p>
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">{t('features.f3.title')}</h3>
                  <p className="text-gray-600">{t('features.f3.desc')}</p>
                </div>
                <div>
                  <h3 className="text-xl font-bold text-gray-900 mb-2">{t('features.f4.title')}</h3>
                  <p className="text-gray-600">{t('features.f4.desc')}</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Workflow Section */}
        <section className="bg-white py-20">
          <div className="max-w-7xl mx-auto px-6">
            <h2 className="text-4xl font-bold text-gray-900 text-center mb-16">{t('workflow.title')}</h2>
            <div className="grid md:grid-cols-2 gap-12 items-center">
              <div className="space-y-8">
                {[1, 2, 3].map((n) => (
                  <div key={n} className="flex items-start space-x-4">
                    <div className="w-12 h-12 bg-teal-600 text-white rounded-full flex items-center justify-center text-xl font-bold flex-shrink-0">
                      {n}
                    </div>
                    <div>
                      <h3 className="text-xl font-bold text-gray-900 mb-2">{t(`workflow.step${n}.title`)}</h3>
                      <p className="text-gray-600">{t(`workflow.step${n}.desc`)}</p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="rounded-2xl overflow-hidden shadow-xl">
                <img
                  src="https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=800&h=600&fit=crop&fm=webp"
                  alt="CopiioAI Workflow Process - Connect API, Add Knowledge, Share as Chat"
                  className="w-full h-full object-cover"
                  loading="lazy"
                  width={800}
                  height={600}
                />
              </div>
            </div>
          </div>
        </section>

        {/* Use Cases Section */}
        <section className="py-20">
          <div className="max-w-7xl mx-auto px-6">
            <h2 className="text-4xl font-bold text-gray-900 text-center mb-16">{t('usecases.title')}</h2>
            <div className="grid md:grid-cols-2 gap-12 items-center">
              <div className="rounded-2xl overflow-hidden shadow-xl">
                <img
                  src="https://images.unsplash.com/photo-1517694712202-14dd9538aa97?w=800&h=600&fit=crop&fm=webp"
                  alt="Developer Coding on Laptop with CopiioAI API Integration"
                  className="w-full h-full object-cover"
                  loading="lazy"
                  width={800}
                  height={600}
                />
              </div>
              <div className="space-y-6">
                {(['c1', 'c2', 'c3', 'c4', 'c5'] as const).map((key) => (
                  <div key={key} className="flex items-start space-x-3">
                    <div className="w-2 h-2 bg-teal-600 rounded-full mt-2 flex-shrink-0" />
                    <p className="text-lg text-gray-700">{t(`usecases.${key}`)}</p>
                  </div>
                ))}
                <p className="text-gray-600 italic pt-4">{t('usecases.note')}</p>
              </div>
            </div>
          </div>
        </section>

        {/* Why CopiioAI Section */}
        <section className="bg-white py-20">
          <div className="max-w-7xl mx-auto px-6">
            <h2 className="text-4xl font-bold text-gray-900 text-center mb-16">{t('why.title')}</h2>
            <div className="grid md:grid-cols-2 gap-12 items-center">
              <div className="space-y-6">
                <p className="text-lg text-gray-700">{t('why.p1')}</p>
                <p className="text-lg font-bold text-gray-900">{t('why.p2')}</p>
                <p className="text-lg text-gray-700">{t('why.p3')}</p>
              </div>
              <div className="rounded-2xl overflow-hidden shadow-xl">
                <img
                  src="https://images.unsplash.com/photo-1522202176988-66273c2fd55f?w=800&h=600&fit=crop&fm=webp"
                  alt="People Collaborating with CopiioAI Chat-based API Tools"
                  className="w-full h-full object-cover"
                  loading="lazy"
                  width={800}
                  height={600}
                />
              </div>
            </div>
          </div>
        </section>

        {/* Final CTA Section */}
        <section className="bg-gradient-to-r from-teal-600 to-teal-700 py-20">
          <div className="max-w-4xl mx-auto px-6 text-center">
            <h2 className="text-4xl font-bold text-white mb-8">{t('cta2.title')}</h2>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a
                href="https://chromewebstore.google.com/detail/copiioai/lejbegpfaanpcilacmakkdediinkmnne"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center space-x-2 px-8 py-4 bg-white text-teal-600 rounded-lg hover:bg-gray-100 transition font-semibold text-lg shadow-lg"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
                <span>{t('cta2.extension')}</span>
              </a>
              <button
                disabled
                className="flex items-center justify-center space-x-2 px-8 py-4 bg-gray-900 text-white rounded-lg font-semibold text-lg shadow-lg cursor-not-allowed opacity-75"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <span>{t('cta2.desktop')}</span>
              </button>
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="bg-gray-100 py-12">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-8">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">{t('footer.support')}</h3>
              <a href="mailto:support@copiioai.com" className="text-teal-600 hover:text-teal-700 text-lg flex items-center justify-center space-x-2">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <span>support@copiioai.com</span>
              </a>
            </div>
            <div className="border-t border-gray-300 pt-8 text-center">
              <div className="flex items-center justify-center space-x-2 mb-4">
                <img src="/logo.png" alt="CopiioAI Logo" className="w-8 h-8 rounded-full" />
                <span className="text-xl font-semibold text-gray-900">CopiioAI</span>
              </div>
              <p className="text-gray-600 mb-4">{t('footer.tagline')}</p>
              <div className="flex justify-center space-x-6 text-sm text-gray-600">
                <a href="/privacy-policy" className="hover:text-teal-600 transition">{t('footer.privacy')}</a>
                <a href="mailto:support@copiioai.com" className="hover:text-teal-600 transition">{t('footer.contact')}</a>
              </div>
              <p className="text-gray-500 text-xs mt-4">{t('footer.copyright')}</p>
              <div className="flex justify-center items-center gap-4 mt-6 flex-wrap">
                <a href="https://fazier.com/launches/copiioai" target="_blank" rel="noopener noreferrer">
                  <img src="https://fazier.com/api/v1/public/badges/embed_image.svg?launch_id=8402&badge_type=daily&theme=light" style={{ height: 40, width: 'auto' }} alt="Fazier badge" />
                </a>
                <a href="https://www.saashub.com/copiioai?utm_source=badge&utm_campaign=badge&utm_content=copiioai&badge_variant=color&badge_kind=approved" target="_blank" rel="noopener noreferrer">
                  <img src="https://cdn-b.saashub.com/img/badges/approved-color.png?v=1" alt="CopiioAI badge" style={{ height: 40, width: 'auto' }} />
                </a>
                <a href="https://www.producthunt.com/products/copiioai?embed=true&utm_source=badge-featured&utm_medium=badge&utm_campaign=badge-copiioai" target="_blank" rel="noopener noreferrer">
                  <img alt="CopiioAI - Natural language interface for accessing internet data | Product Hunt" style={{ height: 40, width: 'auto' }} src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1117523&theme=light&t=1776241061214" />
                </a>
                <a href="https://tinylaunch.com" target="_blank" rel="noopener noreferrer">
                  <img src="https://tinylaunch.com/tinylaunch_badge_launching_soon.svg" alt="TinyLaunch Badge" style={{ height: 40, width: 'auto' }} />
                </a>
                <a href="https://www.uneed.best/tool/copiioai" target="_blank" rel="noopener noreferrer">
                  <img src="https://www.uneed.best/EMBED3.png" alt="Uneed Embed Badge" style={{ height: 40, width: 'auto' }} />
                </a>
                <a href="https://codetrendy.com" target="_blank" rel="noopener noreferrer">
                  <img src="https://codetrendy.com/api/badge?style=classic" alt="Listed on codetrendy.com" style={{ height: 40, width: 'auto' }} />
                </a>
              </div>
            </div>
          </div>
        </footer>
      </main>
    </>
  )
}
