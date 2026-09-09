import { marked } from 'marked'

/**
 * Markdown → rich-text 可渲染 HTML（小程序 <rich-text> 标签子集）。
 * - marked 输出纯字符串（无 DOM 依赖，小程序可用）
 * - 表格降级：<rich-text> 不支持 table 系标签 → 拆成等宽行文本
 */
marked.setOptions({ gfm: true, breaks: true })

const TABLE_ROW_RE = /^\s*\|.*\|\s*$/gm

function demoteTables(md: string): string {
  return md.replace(TABLE_ROW_RE, (row) => {
    const cells = row
      .replace(/^\s*\|/, '')
      .replace(/\|\s*$/, '')
      .split('|')
      .map((c) => c.trim())
      .filter((c) => c && !/^:?-{2,}:?$/.test(c)) // 去掉分隔行
    return cells.length ? cells.join('　·　') : ''
  })
}

export function markdownToHtml(md: string): string {
  if (!md) return ''
  return marked.parse(demoteTables(md), { async: false }) as string
}
