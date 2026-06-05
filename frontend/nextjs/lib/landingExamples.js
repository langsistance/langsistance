export const BILIBILI_USPTO_VIDEO_IDS = ['BV1p2Vz6HEjZ', 'BV1PCVz6WEPW']

export const LANDING_EXAMPLES = [
  {
    slug: 'uspto-china',
    title: '国内用户使用 CopiioAI 检索 USPTO',
    titleKey: 'examples.usptoChina.title',
    href: '/examples/uspto-china',
    summary: '用聊天方式查询美国专利信息、文档、申请人和关键词结果。',
    summaryKey: 'examples.usptoChina.summary',
  },
]

export const LANDING_HEADER_ACTIONS = [
  {
    type: 'link',
    labelKey: 'header.docs',
    fallbackLabel: 'Documentation',
    href: 'https://copiioaicom-spec.github.io/CopiioAI-Natural-language-interface-for-accessing-internet-data/',
    external: true,
  },
  {
    type: 'examples',
    labelKey: 'header.examples',
    fallbackLabel: '使用场景示例',
    items: LANDING_EXAMPLES,
  },
]

export function getLandingExampleBySlug(slug) {
  return LANDING_EXAMPLES.find((example) => example.slug === slug) || null
}

export function getBilibiliEmbedUrl(bvid) {
  const params = new URLSearchParams({
    bvid,
    page: '1',
    high_quality: '1',
    autoplay: '0',
  })
  return `https://player.bilibili.com/player.html?${params.toString()}`
}
