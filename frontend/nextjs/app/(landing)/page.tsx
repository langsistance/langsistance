import type { Metadata } from 'next'
import HomePage from '@/components/app/HomePage'
import LandingSeoContent from '@/components/landing/LandingSeoContent'
import JsonLd, {
  copiioaiOrganizationJsonLd,
  copiioaiWebSiteJsonLd,
  copiioaiSoftwareAppJsonLd,
} from '@/components/JsonLd'

export const metadata: Metadata = {
  // 不设 title: 使用根布局 default（default 不套 "%s | CopiioAI" 模板，
  // 避免 "… | CopiioAI | CopiioAI" 重复)
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
        url: 'https://copiioai.com/og.png',
        width: 1200,
        height: 630,
        alt: 'CopiioAI - AI-Powered Patent Intelligence',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'CopiioAI — AI-Powered Patent Intelligence',
    description:
      'AI-powered patent search, family analysis, and prosecution insights across USPTO, CNIPA, EPO, and more.',
    images: ['https://copiioai.com/og.png'],
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
      {/* SSR 文本内容块 — 让不执行 JS 的爬虫（百度）与 AI 搜索（DeepSeek 等）能读到首页内容 */}
      <LandingSeoContent />
    </>
  )
}
