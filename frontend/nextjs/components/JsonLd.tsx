import Script from 'next/script'

/**
 * Render JSON-LD structured data as a Next.js Script component.
 * Use `strategy="beforeInteractive"` only when data depends on no client state,
 * e.g. static Organization / WebSite / SoftwareApplication.
 */
export default function JsonLd({ data, id }: { data: Record<string, unknown>; id?: string }) {
  return (
    <Script
      id={id || 'jsonld'}
      type="application/ld+json"
      strategy="afterInteractive"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(data) }}
    />
  )
}

/** Organization + WebSite + SoftwareApplication structured data for the landing page */
export function copiioaiOrganizationJsonLd() {
  return {
    '@context': 'https://schema.org',
    '@type': 'Organization',
    name: 'CopiioAI',
    url: 'https://copiioai.com',
    logo: 'https://copiioai.com/logo.png',
    sameAs: [
      'https://chromewebstore.google.com/detail/copiioai/lejbegpfaanpcilacmakkdediinkmnne',
      'https://www.producthunt.com/products/copiioai',
    ],
    contactPoint: {
      '@type': 'ContactPoint',
      email: 'support@copiioai.com',
      contactType: 'customer support',
    },
  }
}

export function copiioaiWebSiteJsonLd() {
  return {
    '@context': 'https://schema.org',
    '@type': 'WebSite',
    name: 'CopiioAI',
    url: 'https://copiioai.com',
    description:
      'AI-powered patent search, family analysis, and prosecution insights across USPTO, CNIPA, EPO, and more. Chat with patent data using natural language.',
    inLanguage: ['en', 'zh-Hans', 'ja', 'ko', 'es', 'fr', 'de'],
    potentialAction: {
      '@type': 'SearchAction',
      target: {
        '@type': 'EntryPoint',
        urlTemplate: 'https://copiioai.com/search?q={search_term_string}',
      },
      'query-input': 'required name=search_term_string',
    },
  }
}

export function copiioaiSoftwareAppJsonLd() {
  return {
    '@context': 'https://schema.org',
    '@type': 'SoftwareApplication',
    name: 'CopiioAI',
    applicationCategory: 'BusinessApplication',
    operatingSystem: 'Chrome',
    offers: {
      '@type': 'Offer',
      price: '0',
      priceCurrency: 'USD',
    },
    url: 'https://copiioai.com',
    description:
      'AI-powered patent intelligence platform. Search, analyze, and compare patents across USPTO, CNIPA, EPO, and more using natural language AI.',
  }
}
