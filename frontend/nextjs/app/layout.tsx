import type { Metadata } from 'next'
import Script from 'next/script'

export const metadata: Metadata = {
  title: {
    default: 'CopiioAI — AI-Powered API Tool Builder',
    template: '%s | CopiioAI',
  },
  description: 'Build AI-powered tools from your APIs. CopiioAI converts browser-captured API requests into conversational AI tools.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
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
