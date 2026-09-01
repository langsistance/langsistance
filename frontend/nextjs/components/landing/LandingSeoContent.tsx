/**
 * Server-rendered SEO content block for the homepage.
 *
 * The chat app itself is a client component — without this block the SSR
 * HTML carries no indexable text, so non-JS crawlers (Baidu) and AI search
 * engines (DeepSeek, etc.) see an empty page.  Static bilingual copy
 * rendered below the app: visible to users at the footer, readable by
 * every crawler in the raw HTML.
 */
export default function LandingSeoContent() {
  return (
    <section
      aria-label="About CopiioAI"
      className="w-full border-t border-gray-100 bg-white py-12"
    >
      <div className="mx-auto max-w-3xl px-6">
        <h2 className="text-lg font-semibold text-gray-900">
          CopiioAI — AI 专利情报检索与分析平台
        </h2>
        <p className="mt-3 text-sm leading-6 text-gray-600">
          CopiioAI 是一款面向专利检索与分析的 AI 助手。你可以在对话中直接描述
          技术方案、粘贴产品链接、上传专利文件或输入专利号，系统会自动检索相关
          专利并给出结构化分析结果。无需学习专利数据库的检索语法，也无需在多个
          官方数据库之间切换。
        </p>
        <p className="mt-3 text-sm leading-6 text-gray-600">
          CopiioAI is an AI assistant for patent search and analysis. Describe a
          technology, paste a product link, upload a patent document, or enter a
          patent number in plain language — it searches and analyzes relevant
          patents for you, no database query syntax required.
        </p>

        <h3 className="mt-8 text-sm font-semibold text-gray-900">核心功能</h3>
        <ul className="mt-2 list-disc pl-5 text-sm leading-6 text-gray-600">
          <li>专利检索：支持美国专利商标局（USPTO）与中国专利（CNIPA）双源检索，按技术主题、关键词、申请人或专利号查询</li>
          <li>专利查重：上传专利文件，自动比对现有技术，判断重复与新颖性风险</li>
          <li>审查历史分析：分析美国、中国、欧洲、日本专利的审查历史与审查意见</li>
          <li>同族分析：追踪同一发明在多个国家的专利家族分布</li>
          <li>报告生成：自动生成可下载的分析报告（PDF / Word）</li>
          <li>Chrome 扩展：浏览网页时随手查询与抓取专利信息</li>
        </ul>

        <h3 className="mt-8 text-sm font-semibold text-gray-900">支持的数据源</h3>
        <p className="mt-2 text-sm leading-6 text-gray-600">
          美国专利商标局 USPTO（申请、授权、公开号）、中国国家知识产权局 CNIPA
          专利（含中文全文检索）、欧洲专利局 EPO、日本特许厅 JPO。覆盖发明、
          实用新型、外观设计等专利类型。
        </p>

        <h3 className="mt-8 text-sm font-semibold text-gray-900">使用方式</h3>
        <ul className="mt-2 list-disc pl-5 text-sm leading-6 text-gray-600">
          <li>直接提问：例如描述一项技术方案，或输入专利号（US、CN 格式均可）</li>
          <li>上传文件：上传专利说明书、权利要求书或产品文档进行比对分析</li>
          <li>粘贴链接：粘贴产品页面或电商链接，自动解析并检索对应专利</li>
          <li>批量任务：批量检索专利号、批量下载专利文件、生成侵权对比分析</li>
        </ul>

        <p className="mt-8 text-xs leading-5 text-gray-400">
          CopiioAI — AI-Powered Patent Intelligence. Patent search, family
          analysis, prosecution insights across USPTO, CNIPA, EPO and more.
        </p>
      </div>
    </section>
  )
}
