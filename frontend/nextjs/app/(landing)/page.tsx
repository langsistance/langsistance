import type { Metadata } from 'next'
import LandingPage from '@/components/landing/LandingPage'

export const metadata: Metadata = {
  title: 'CopiioAI — Turn Any API Into a Conversational Interface',
  description:
    'CopiioAI turns any API into a chat-based interface. Perfect for both developers and non-developers - build and share chat tools powered by your APIs so anyone can access real-time data with natural language, no coding or frontend required.',
  keywords: [
    'API', 'chat interface', 'conversational AI', 'API tools', 'no-code API',
    'chat-based API', 'API to chat', 'developer tools', 'AI assistant', 'natural language API',
  ],
  openGraph: {
    title: 'CopiioAI — Turn Any API Into a Conversational Interface',
    description:
      'CopiioAI turns any API into a chat-based interface. Build and share chat tools powered by your APIs.',
    url: 'https://copiioai.com',
    siteName: 'CopiioAI',
    type: 'website',
    images: [
      {
        url: 'https://copiioai.com/og-image.png',
        width: 1200,
        height: 630,
        alt: 'CopiioAI - AI-Powered API Tool Builder',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'CopiioAI — Turn Any API Into a Conversational Interface',
    description:
      'CopiioAI turns any API into a chat-based interface. Build and share chat tools powered by your APIs.',
    images: ['https://copiioai.com/og-image.png'],
  },
  alternates: {
    canonical: 'https://copiioai.com',
  },
}

export default function Home() {
  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'Organization',
            name: 'CopiioAI',
            url: 'https://copiioai.com',
            logo: 'https://copiioai.com/logo.png',
            description:
              'CopiioAI turns any API into a chat-based interface. Perfect for both developers and non-developers.',
            sameAs: [
              'https://chromewebstore.google.com/detail/copiioai/lejbegpfaanpcilacmakkdediinkmnne',
            ],
          }),
        }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'WebSite',
            name: 'CopiioAI',
            url: 'https://copiioai.com',
            potentialAction: {
              '@type': 'SearchAction',
              target: 'https://copiioai.com/search?q={search_term_string}',
              'query-input': 'required name=search_term_string',
            },
          }),
        }}
      />
      <LandingPage />
    </>
  )
}
