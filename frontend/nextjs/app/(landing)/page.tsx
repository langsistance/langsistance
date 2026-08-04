import type { Metadata } from 'next'
import HomePage from '@/components/app/HomePage'
import JsonLd, {
  copiioaiOrganizationJsonLd,
  copiioaiWebSiteJsonLd,
  copiioaiSoftwareAppJsonLd,
} from '@/components/JsonLd'

export const metadata: Metadata = {
  title: 'CopiioAI — AI-Powered Patent Intelligence',
  description:
    'AI-powered patent search, family analysis, and prosecution insights. Query patents from USPTO, CNIPA, EPO, and more using natural language — no manual database search required.',
  keywords: [
    'patent search', 'patent AI', 'patent analysis', 'patent family', 'USPTO', 'CNIPA',
    'EPO patent', 'IP intelligence', 'patent prosecution', 'cross-jurisdiction patent',
  ],
  openGraph: {
    title: 'CopiioAI — AI-Powered Patent Intelligence',
    description:
      'AI-powered patent search, family analysis, and prosecution insights across USPTO, CNIPA, EPO, and more.',
    url: 'https://copiioai.com',
    siteName: 'CopiioAI',
    type: 'website',
    images: [
      {
        url: 'https://copiioai.com/icon.png',
        width: 512,
        height: 512,
        alt: 'CopiioAI - AI-Powered Patent Intelligence',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'CopiioAI — AI-Powered Patent Intelligence',
    description:
      'AI-powered patent search, family analysis, and prosecution insights across USPTO, CNIPA, EPO, and more.',
    images: ['https://copiioai.com/icon.png'],
  },
  alternates: {
    canonical: 'https://copiioai.com',
  },
}

export default function Home() {
  return (
    <>
      <JsonLd id="jsonld-org" data={copiioaiOrganizationJsonLd()} />
      <JsonLd id="jsonld-website" data={copiioaiWebSiteJsonLd()} />
      <JsonLd id="jsonld-app" data={copiioaiSoftwareAppJsonLd()} />
      <HomePage />
    </>
  )
}
