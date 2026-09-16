import type { Metadata } from 'next'
import JsonLd from '@/components/JsonLd'

const LAST_UPDATED = '2026-09-16'
const VERSION = '1.0'

export const metadata: Metadata = {
  title: 'Information Security',
  description:
    'How CopiioAI protects your patent data — no AI training on your content, account-isolated conversations, and uploads deleted after processing.',
  keywords: [
    'patent data security', 'AI patent confidentiality', 'CopiioAI security',
    'patent search privacy', 'IP data protection',
  ],
  openGraph: {
    title: 'Information Security | CopiioAI',
    description:
      'No AI training on your content, account-isolated conversations, uploads deleted after processing.',
    url: 'https://copiioai.com/security',
    siteName: 'CopiioAI',
    type: 'website',
  },
  alternates: {
    canonical: 'https://copiioai.com/security',
  },
}

const COPY = {
  zh: {
    langLabel: '中文',
    h1: '信息安全',
    intro: '专利是您最重要的资产之一。以下是我们对您数据的承诺，以及每条承诺对应的具体做法。',
    promises: [
      {
        title: '不外传 —— 不用于训练任何 AI 模型',
        body: '您的提问、上传文件与分析结果，不会被用于训练任何 AI 模型。模型推理通过企业级 API 通道完成，该通道默认不保留数据用于训练用途。',
      },
      {
        title: '不外泄 —— 会话按账号隔离',
        body: '对话记录按账号严格隔离，每次读取均按账号过滤。上传文件仅您本人可访问。',
      },
      {
        title: '不外流 —— 上传文件处理后删除',
        body: '上传的专利文件在分析处理完成后从服务器删除。我们不会将其用于任何其他用途。',
      },
    ],
    transportTitle: '传输与存储',
    transportItems: [
      '所有请求经 HTTPS 加密传输。',
      '登录密码以 AES-GCM 加密后传输，服务端不落明文。',
    ],
    logsTitle: '日志',
    logsBody: '用于诊断的日志中，提问内容仅保留前 80 个字符。',
    footerNote: '我们会持续加强数据保护措施，本页随能力升级更新。',
    versionLine: `Version ${VERSION} · Last updated ${LAST_UPDATED}`,
  },
  en: {
    langLabel: 'English',
    h1: 'Information Security',
    intro: 'Patents are among your most valuable assets. Here is what we commit to — and how each commitment is implemented.',
    promises: [
      {
        title: 'Not shared — never used to train AI models',
        body: 'Your prompts, uploaded files, and analysis results are never used to train any AI model. Model inference runs through enterprise API channels, which by default do not retain data for training.',
      },
      {
        title: 'Not exposed — conversations isolated by account',
        body: 'Conversation records are strictly isolated by account, and every read is filtered by account. Uploaded files are accessible only to you.',
      },
      {
        title: 'Not retained — uploads deleted after processing',
        body: 'Uploaded patent files are deleted from our servers once analysis finishes. We do not use them for any other purpose.',
      },
    ],
    transportTitle: 'Transport and storage',
    transportItems: [
      'All requests are transmitted over HTTPS.',
      'Login passwords are encrypted with AES-GCM in transit; no plaintext is stored server-side.',
    ],
    logsTitle: 'Logging',
    logsBody: 'In diagnostic logs, prompt content is truncated to the first 80 characters.',
    footerNote: 'We continue to strengthen our data protection measures. This page is updated as capabilities improve.',
    versionLine: `Version ${VERSION} · Last updated ${LAST_UPDATED}`,
  },
} as const

type CopyLang = keyof typeof COPY

function SecurityCopy({ copy }: { copy: (typeof COPY)[CopyLang] }) {
  return (
    <section className="mb-14">
      <h2 className="text-2xl font-bold text-gray-900 mb-4">{copy.h1}</h2>
      <p className="text-gray-700 mb-8">{copy.intro}</p>

      <div className="space-y-6">
        {copy.promises.map((p) => (
          <div key={p.title} className="border-l-2 border-teal-600 pl-5">
            <h3 className="text-lg font-semibold text-teal-700 mb-2">{p.title}</h3>
            <p className="text-gray-700">{p.body}</p>
          </div>
        ))}
      </div>

      <h3 className="text-xl font-bold text-gray-900 mt-10 mb-3">{copy.transportTitle}</h3>
      <ul className="list-disc pl-6 text-gray-700 space-y-1">
        {copy.transportItems.map((t) => (
          <li key={t}>{t}</li>
        ))}
      </ul>

      <h3 className="text-xl font-bold text-gray-900 mt-10 mb-3">{copy.logsTitle}</h3>
      <p className="text-gray-700">{copy.logsBody}</p>

      <p className="text-gray-500 text-sm mt-10">{copy.footerNote}</p>
      <p className="text-gray-400 text-xs mt-2">{copy.versionLine}</p>
    </section>
  )
}

export default function SecurityPage() {
  return (
    <>
      <JsonLd
        id="jsonld-security"
        data={{
          '@context': 'https://schema.org',
          '@type': 'WebPage',
          name: 'Information Security',
          description:
            'How CopiioAI protects your patent data — no AI training on your content, account-isolated conversations, and uploads deleted after processing.',
          publisher: {
            '@type': 'Organization',
            name: 'CopiioAI',
            url: 'https://copiioai.com',
          },
        }}
      />
      <div className="max-w-3xl mx-auto px-6 py-16">
        <nav className="mb-5 text-sm text-gray-500">
          <a href="/" className="text-teal-600 hover:underline">CopiioAI</a>
          {' / '}
          <span>Information Security</span>
        </nav>

        <h1 className="text-3xl font-bold text-gray-900 mb-10">Information Security</h1>

        <SecurityCopy copy={COPY.zh} />
        <div className="border-t border-gray-200 pt-4" />
        <SecurityCopy copy={COPY.en} />

        <div className="mt-12 text-sm">
          <a href="/privacy-policy" className="text-teal-600 hover:underline">Privacy Policy</a>
        </div>
      </div>
    </>
  )
}
