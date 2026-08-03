import type { Metadata } from 'next'
import Script from 'next/script'

export const metadata: Metadata = {
  metadataBase: new URL('https://copiioai.com'),
  title: {
    default: 'CopiioAI — AI-Powered Patent Intelligence',
    template: '%s | CopiioAI',
  },
  description:
    'AI-powered patent search, family analysis, and prosecution insights across USPTO, CNIPA, EPO, and more. Chat with patent data — no manual search required.',
  applicationName: 'CopiioAI',
  creator: 'CopiioAI',
  publisher: 'CopiioAI',
  keywords: [
    'patent search', 'patent analysis', 'AI patent', 'patent family analysis',
    'patent prosecution', 'USPTO search', 'CNIPA', 'EPO patent', 'IP intelligence',
    'patent AI', 'cross-jurisdiction patent', 'patent document analysis',
    'Chrome extension', 'AI assistant',
  ],
  robots: {
    index: true,
    follow: true,
    'max-snippet': -1,
    'max-image-preview': 'large',
    'max-video-preview': -1,
  },
  openGraph: {
    type: 'website',
    siteName: 'CopiioAI',
    locale: 'en_US',
  },
  icons: {
    icon: '/icon.png',
    apple: '/icon.png',
  },
  manifest: '/manifest.webmanifest',
  alternates: {
    canonical: 'https://copiioai.com',
    languages: {
      'en': 'https://copiioai.com',
      'zh-Hans': 'https://copiioai.com',
      'ja': 'https://copiioai.com',
      'ko': 'https://copiioai.com',
      'es': 'https://copiioai.com',
      'fr': 'https://copiioai.com',
      'de': 'https://copiioai.com',
    },
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        {/* Preconnect to external origins for faster resource loading */}
        <link rel="preconnect" href="https://www.googletagmanager.com" />
        <link rel="preconnect" href="https://images.unsplash.com" />
        <link rel="dns-prefetch" href="https://www.youtube.com" />
        <link rel="dns-prefetch" href="https://img.youtube.com" />
      </head>
      <body>
        <Script
          src="https://www.googletagmanager.com/gtag/js?id=G-LLPQHRD2EZ"
          strategy="afterInteractive"
        />
        <Script id="google-analytics" strategy="afterInteractive">
          {`
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());
            gtag('config', 'G-LLPQHRD2EZ');
          `}
        </Script>
        {children}
      </body>
    </html>
  )
}
