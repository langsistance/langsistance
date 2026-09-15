/**
 * 分享载荷：标题与配图固定为品牌，不携带任何会话内容。
 *
 * 本模块**刻意不 import 任何资源** —— 它要能被 `node --test` 直接加载。
 * tsconfig.test.json 只编译不拷资源，若此处 import 图片，编译产物里的
 * require('../assets/share-card.png') 会指向不会生成的 dist-test/assets/，
 * 运行期直接 MODULE_NOT_FOUND（类型层面是合法的，见 types/assets.d.ts）。
 * 故配图路径由调用方（页面）import 后作为参数传入。
 *
 * 设计依据：docs/superpowers/specs/2026-09-15-weapp-landing-share-design.md
 */

export const SHARE_TITLE = 'CopiioAI 专利情报，一问即得'
export const SHARE_PATH = '/pages/chat/index'

/** 品牌卡片载荷：标题与配图固定，不携带任何会话内容。 */
export function buildShareCard(imagePath: string) {
  return { title: SHARE_TITLE, imageUrl: imagePath }
}
