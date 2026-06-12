import type { Metadata } from 'next'
import HomePage from '@/components/app/HomePage'

export const metadata: Metadata = {
  title: 'CopiioAI — AI-Powered API Tool Builder',
  description:
    'CopiioAI turns any API into a chat-based interface. Build and share chat tools powered by your APIs so anyone can access real-time data with natural language.',
  keywords: [
    'API', 'chat interface', 'conversational AI', 'API tools', 'no-code API',
    'chat-based API', 'API to chat', 'developer tools', 'AI assistant', 'natural language API',
  ],
  openGraph: {
    title: 'CopiioAI — AI-Powered API Tool Builder',
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
    title: 'CopiioAI — AI-Powered API Tool Builder',
    description:
      'CopiioAI turns any API into a chat-based interface. Build and share chat tools powered by your APIs.',
    images: ['https://copiioai.com/og-image.png'],
  },
  alternates: {
    canonical: 'https://copiioai.com',
  },
}

export default function Home() {
  return <HomePage />
}
