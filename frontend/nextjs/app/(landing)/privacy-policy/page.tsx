import type { Metadata } from 'next'
import JsonLd from '@/components/JsonLd'

const LAST_UPDATED = '2026-09-16'

export const metadata: Metadata = {
  title: 'Privacy Policy',
  description:
    'Privacy Policy for CopiioAI — how we handle patent search queries, uploaded documents, analysis results, and account data.',
  keywords: [
    'privacy policy', 'CopiioAI privacy', 'patent data protection', 'AI patent privacy',
  ],
  openGraph: {
    title: 'Privacy Policy | CopiioAI',
    description:
      'How CopiioAI handles patent search queries, uploaded documents, analysis results, and account data.',
    url: 'https://copiioai.com/privacy-policy',
    siteName: 'CopiioAI',
    type: 'website',
  },
  alternates: {
    canonical: 'https://copiioai.com/privacy-policy',
  },
}

const COPY = {
  zh: {
    h1: 'CopiioAI 隐私政策',
    intro:
      'CopiioAI 是面向专利检索与分析的 AI 工具。我们尊重您的隐私，并以透明、负责的方式处理数据。本政策说明我们收集哪些数据、如何使用、以及您拥有哪些控制权。',
    sections: [
      {
        h: '1. 我们处理的数据',
        items: [
          '检索提问：您输入的自然语言问题与技术描述。',
          '上传文件：您主动上传的专利说明书、权利要求、审查文件等（PDF / DOCX / XML）。',
          '分析结果：系统为您生成的检索结果、分析与报告。',
          '知识库内容：您在本产品内创建或保存的条目。',
          '账号信息：邮箱地址，以及用于身份验证的令牌。',
          '匿名使用统计：页面访问与功能使用情况、设备与浏览器类型、国家/地区级地理位置。',
        ],
      },
      {
        h: '2. 我们如何使用数据',
        items: [
          '提供并运行 CopiioAI 的核心功能（检索、分析、下载）。',
          '在您的账号下保存对话记录，以便您随时回看。',
          '改进产品可靠性与使用体验。',
        ],
        note: '我们不会将您的数据用于广告、追踪或用户画像。匿名统计仅用于产品改进。用于诊断的日志中，提问内容仅保留前 80 个字符。',
      },
      {
        h: '3. AI 训练政策',
        p: '您的提问、上传文件与分析结果不会被用于训练任何 AI 模型。模型推理通过企业级 API 通道完成，该通道默认不保留数据用于训练用途。',
      },
      {
        h: '4. 数据安全',
        items: [
          '所有请求经 HTTPS 加密传输。',
          '登录密码以 AES-GCM 加密后传输，服务端不落明文。',
          '对话记录按账号严格隔离，每次读取均按账号过滤。',
          '上传的专利文件在分析处理完成后从服务器删除。',
        ],
        note: '关于每条安全措施的具体说明，请见信息安全页。',
      },
      {
        h: '5. 数据共享',
        p: '我们不会出售或出租您的数据。数据仅在以下情况被处理：为提供核心服务功能而进行的技术处理；您主动选择分享知识库条目时；法律法规要求时。',
      },
      {
        h: '6. 数据留存与您的控制',
        items: [
          '您的对话记录与知识库条目会保留至您主动删除，或通过下方邮箱请求删除账号数据为止。',
          '您可以在产品内随时删除自己的对话记录与知识库条目。',
          '您也可以通过下方邮箱联系我们，请求删除账号数据。',
        ],
      },
      {
        h: '7. 本政策的更新',
        p: '我们可能不定期更新本政策。任何变更都会在本页顶部的更新日期中体现。',
      },
    ],
    contactH: '8. 联系我们',
    contactP: '如对本隐私政策有疑问，请联系：',
    securityLink: '查看信息安全页',
  },
  en: {
    h1: 'Privacy Policy for CopiioAI',
    intro:
      'CopiioAI is an AI-powered patent search and analysis tool. We respect your privacy and handle data transparently and responsibly. This policy explains what we collect, how we use it, and what control you have.',
    sections: [
      {
        h: '1. Data We Process',
        items: [
          'Search queries: the natural-language questions and technical descriptions you enter.',
          'Uploaded files: patent specifications, claims, and prosecution documents you choose to upload (PDF / DOCX / XML).',
          'Analysis results: search results, analyses, and reports generated for you.',
          'Knowledge base entries: items you create or save in the product.',
          'Account information: your email address and an identity token used for authentication.',
          'Anonymous usage statistics: page views and feature usage, device and browser type, country/region-level location.',
        ],
      },
      {
        h: '2. How We Use Data',
        items: [
          'To provide and operate CopiioAI core features (search, analysis, download).',
          'To store conversation records under your account so you can revisit them.',
          'To improve product reliability and user experience.',
        ],
        note: 'We do not use your data for advertising, tracking, or profiling. Anonymous statistics are used solely for product improvement. In diagnostic logs, prompt content is truncated to the first 80 characters.',
      },
      {
        h: '3. AI Training Policy',
        p: 'Your prompts, uploaded files, and analysis results are never used to train any AI model. Model inference runs through enterprise API channels, which by default do not retain data for training purposes.',
      },
      {
        h: '4. Data Security',
        items: [
          'All requests are transmitted over HTTPS.',
          'Login passwords are encrypted with AES-GCM in transit; no plaintext is stored server-side.',
          'Conversation records are strictly isolated by account, and every read is filtered by account.',
          'Uploaded patent files are deleted from our servers once analysis finishes.',
        ],
        note: 'For a detailed explanation of each measure, see our Information Security page.',
      },
      {
        h: '5. Data Sharing',
        p: 'We do not sell or rent your data. Data is processed only to provide core service functionality, when you choose to share a knowledge base entry, or when required by law.',
      },
      {
        h: '6. Retention and Your Control',
        items: [
          'Your conversation records and knowledge base entries are retained until you delete them, or until you request deletion of your account data via the email address below.',
          'You may delete your conversation records and knowledge base entries within the product.',
          'You may contact us at the address below to request deletion of your account data.',
        ],
      },
      {
        h: '7. Changes to This Policy',
        p: 'We may update this policy periodically. Any changes will be reflected in the "Last updated" date at the top of this page.',
      },
    ],
    contactH: '8. Contact',
    contactP: 'If you have questions about this Privacy Policy, please contact:',
    securityLink: 'View our Information Security page',
  },
} as const

type CopyLang = keyof typeof COPY

function PolicyCopy({ copy }: { copy: (typeof COPY)[CopyLang] }) {
  return (
    <section className="mb-14">
      <h2 className="text-2xl font-bold text-gray-900 mb-4">{copy.h1}</h2>
      <p className="text-gray-700 mb-8">{copy.intro}</p>

      {copy.sections.map((s) => (
        <div key={s.h}>
          <h3 className="text-xl font-bold text-gray-900 mt-8 mb-3">{s.h}</h3>
          {'p' in s && <p className="text-gray-700 mb-4">{s.p}</p>}
          {'items' in s && (
            <ul className="list-disc pl-6 text-gray-700 mb-4 space-y-1">
              {s.items.map((i) => (
                <li key={i}>{i}</li>
              ))}
            </ul>
          )}
          {'note' in s && <p className="text-gray-600 mb-4">{s.note}</p>}
        </div>
      ))}

      <h3 className="text-xl font-bold text-gray-900 mt-8 mb-3">{copy.contactH}</h3>
      <p className="text-gray-700 mb-2">{copy.contactP}</p>
      <p className="text-gray-700">
        <strong>Email:</strong> copiioai.com@gmail.com
      </p>
    </section>
  )
}

export default function PrivacyPolicy() {
  return (
    <>
      <JsonLd
        id="jsonld-webpage"
        data={{
          '@context': 'https://schema.org',
          '@type': 'WebPage',
          name: 'Privacy Policy',
          description:
            'Privacy Policy for CopiioAI — how we handle patent search queries, uploaded documents, analysis results, and account data.',
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
          <span>Privacy Policy</span>
        </nav>

        <h1 className="text-3xl font-bold text-gray-900 mb-2">Privacy Policy</h1>
        <p className="text-gray-500 mb-10">Last updated: {LAST_UPDATED}.</p>

        <PolicyCopy copy={COPY.zh} />
        <div className="border-t border-gray-200 pt-4" />
        <PolicyCopy copy={COPY.en} />

        <div className="mt-12 text-sm">
          <a href="/security" className="text-teal-600 hover:underline">
            {COPY.zh.securityLink} / {COPY.en.securityLink}
          </a>
        </div>
      </div>
    </>
  )
}
