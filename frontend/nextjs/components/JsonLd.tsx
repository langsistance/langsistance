/**
 * Render JSON-LD structured data as a plain inline <script> tag.
 *
 * Must NOT use next/script: its `beforeInteractive` strategy does not emit a
 * real <script type="application/ld+json"> tag into the HTML under
 * `output: 'export'` (static export) — the data only lands in client-side
 * loader instructions, invisible to crawlers that don't execute JS (Baidu)
 * and AI search engines (DeepSeek, Perplexity, etc.).
 *
 * This file is imported only from server components, so the plain <script>
 * tag is rendered into the exported HTML directly. JSON-LD is inert markup
 * (`type="application/ld+json"` never executes), so dangerouslySetInnerHTML
 * is safe here — the content is our own static data, never user input.
 */
export default function JsonLd({ data, id }: { data: Record<string, unknown>; id?: string }) {
  return (
    <script
      id={id || 'jsonld'}
      type="application/ld+json"
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
    // No SearchAction: the /search?q= route this previously pointed to
    // does not exist — a dead structured-data link is worse than none.
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
