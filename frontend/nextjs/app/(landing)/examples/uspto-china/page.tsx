import type { Metadata } from 'next'
import LandingHeader from '@/components/landing/LandingHeader'
import {
  BILIBILI_USPTO_VIDEO_IDS,
  getBilibiliEmbedUrl,
  getLandingExampleBySlug,
} from '@/lib/landingExamples'

const example = getLandingExampleBySlug('uspto-china')

export const metadata: Metadata = {
  title: '国内用户使用 CopiioAI 检索 USPTO',
  description:
    '一个面向国内美国专利检索场景的 CopiioAI 使用示例：通过自然语言查询美国专利信息、专利文档、关键词和被转让人结果。',
  alternates: {
    canonical: 'https://app-cn.copiioai.com/examples/uspto-china',
  },
}

function BilibiliVideo({ bvid, title }: { bvid: string; title: string }) {
  return (
    <div className="bg-white rounded-xl shadow-xl overflow-hidden border border-gray-100">
      <div className="aspect-video">
        <iframe
          className="w-full h-full"
          src={getBilibiliEmbedUrl(bvid)}
          title={title}
          allow="fullscreen; encrypted-media; picture-in-picture"
          allowFullScreen
        />
      </div>
    </div>
  )
}

export default function UsptoChinaExamplePage() {
  return (
    <>
      <LandingHeader />
      <main className="pt-20 bg-gray-50">
        <section className="bg-white">
          <div className="max-w-6xl mx-auto px-6 py-16 md:py-20">
            <a href="/" className="text-sm font-semibold text-teal-700 hover:text-teal-800">
              返回首页
            </a>
            <div className="mt-8 grid lg:grid-cols-[1.15fr_0.85fr] gap-12 items-center">
              <div>
                <p className="text-sm font-semibold tracking-wide text-teal-700 uppercase">
                  使用场景示例
                </p>
                <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mt-4 leading-tight">
                  {example?.title || '国内用户使用 CopiioAI 检索 USPTO'}
                </h1>
                <p className="text-xl text-gray-600 mt-6 leading-8">
                  对国内用户来说，美国专利检索往往不只是查一个号码。真正耗时的是在多个页面之间切换、
                  从不同结果里提取字段、再把信息整理成能直接回答客户问题的结论。CopiioAI 希望把这个过程
                  变成一次自然语言对话。
                </p>
                <div className="mt-8 flex flex-col sm:flex-row gap-4">
                  <a
                    href="/app/chat"
                    className="inline-flex items-center justify-center px-6 py-3 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition font-semibold"
                  >
                    打开 CopiioAI
                  </a>
                  <a
                    href="#videos"
                    className="inline-flex items-center justify-center px-6 py-3 border border-gray-300 text-gray-800 rounded-lg hover:border-teal-600 hover:text-teal-700 transition font-semibold"
                  >
                    查看演示视频
                  </a>
                </div>
              </div>
              <div className="bg-gray-900 text-white rounded-xl p-8 shadow-xl">
                <h2 className="text-2xl font-bold mb-6">适合这类工作流</h2>
                <ul className="space-y-4 text-gray-200">
                  <li>一个专利要打开多个 USPTO 页面才能确认完整信息。</li>
                  <li>需要从公开号、授权号或申请号继续追查文档。</li>
                  <li>客户临时提问时，不想在多个检索系统之间反复切换。</li>
                  <li>查询结果还需要整理、摘要和转成可沟通的文字。</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        <section className="py-16">
          <div className="max-w-6xl mx-auto px-6">
            <div className="grid md:grid-cols-2 gap-10">
              <div>
                <h2 className="text-3xl font-bold text-gray-900 mb-5">为什么做这个示例</h2>
                <div className="space-y-5 text-lg text-gray-700 leading-8">
                  <p>
                    近几年做美国专利相关工作时，我经常遇到同一个问题：USPTO 数据是权威的，
                    但页面和接口并不总是适合临时问答。一个简单问题可能要先查专利基本信息，
                    再找到申请号，再去查文件，再把结果整理成客户能看懂的结论。
                  </p>
                  <p>
                    CopiioAI 的思路不是再做一个传统检索页面，而是把常用检索能力配置成聊天工具。
                    用户可以直接说“我想查询这个公开号的全部专利文档”，系统再按预设知识和 API 步骤去查询。
                  </p>
                </div>
              </div>
              <div>
                <h2 className="text-3xl font-bold text-gray-900 mb-5">目前支持的美国专利查询</h2>
                <ul className="space-y-4 text-lg text-gray-700">
                  <li>通过专利申请号、授权号、公开号查询美国专利信息。</li>
                  <li>通过专利申请号、授权号、公开号查询美国专利文档。</li>
                  <li>通过关键字查询美国专利信息。</li>
                  <li>通过被转让人查询美国专利信息。</li>
                  <li>通过关键字查询谷歌收录的专利。</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        <section className="bg-white py-16">
          <div className="max-w-6xl mx-auto px-6">
            <h2 className="text-3xl font-bold text-gray-900 mb-6">可以怎么问</h2>
            <div className="grid md:grid-cols-3 gap-6">
              {[
                'I want to obtain all patent documents with the patent publication number US20250014493A1.',
                'Search for patents assigned to a specific company and summarize the results.',
                'Find US patent information by keyword and list the most relevant records.',
              ].map((prompt) => (
                <div key={prompt} className="bg-gray-50 border border-gray-100 rounded-xl p-6">
                  <p className="text-gray-800 leading-7">{prompt}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="videos" className="py-16">
          <div className="max-w-6xl mx-auto px-6">
            <div className="mb-8">
              <h2 className="text-3xl font-bold text-gray-900">演示视频</h2>
              <p className="text-lg text-gray-600 mt-3">
                下面两个视频展示了 CopiioAI 在美国专利检索场景中的实际使用方式。
              </p>
            </div>
            <div className="grid lg:grid-cols-2 gap-8">
              <BilibiliVideo bvid={BILIBILI_USPTO_VIDEO_IDS[0]} title="CopiioAI USPTO patent search demo 1" />
              <BilibiliVideo bvid={BILIBILI_USPTO_VIDEO_IDS[1]} title="CopiioAI USPTO patent search demo 2" />
            </div>
          </div>
        </section>

        <section className="bg-teal-700 py-16">
          <div className="max-w-4xl mx-auto px-6 text-center">
            <h2 className="text-3xl font-bold text-white mb-5">把重复检索流程交给 CopiioAI</h2>
            <p className="text-lg text-teal-50 leading-8 mb-8">
              如果你的工作也经常需要查 USPTO、整理专利字段、下载文档或快速回答客户问题，
              可以先从这个示例开始，把高频查询变成可复用的聊天能力。
            </p>
            <a
              href="/app/chat"
              className="inline-flex items-center justify-center px-8 py-4 bg-white text-teal-700 rounded-lg hover:bg-gray-100 transition font-semibold text-lg"
            >
              开始使用
            </a>
          </div>
        </section>
      </main>
    </>
  )
}
