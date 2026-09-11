import { marked } from 'marked'

/**
 * Markdown → 可渲染片段（小程序 <rich-text> 标签子集）。
 *
 * marked 输出纯字符串（无 DOM 依赖，小程序可用）；但 <rich-text> 不支持
 * table 系标签，早期做法是把表格拍平成一串 `　·　` 连接的文本 —— 专利检索的
 * 回答主体恰恰是一张结果表，拍平后完全没法看。
 *
 * 现在改为把 markdown 拆成片段：非表格部分走 rich-text（保留标题/列表/粗体/
 * 代码块等），表格部分解析成行列数据交给页面用原生 View 渲染成卡片列表。
 */
marked.setOptions({ gfm: true, breaks: true })

export interface MdTable {
  headers: string[]
  /** 已按表头列数对齐补空，长度与 headers 一致 */
  rows: string[][]
}

export type MdSegment =
  | { kind: 'html'; html: string }
  | { kind: 'table'; table: MdTable }

const TABLE_LINE_RE = /^\s*\|.*\|\s*$/

/** `| a | b |` → ['a', 'b'] */
function splitRow(line: string): string[] {
  return line
    .replace(/^\s*\|/, '')
    .replace(/\|\s*$/, '')
    .split('|')
    .map((c) => c.trim())
}

/** 表头下的 `|---|---|` 分隔行 */
function isSeparatorRow(cells: string[]): boolean {
  return cells.length > 0 && cells.every((c) => /^:?-{2,}:?$/.test(c))
}

function renderHtml(md: string): string {
  return marked.parse(md, { async: false }) as string
}

/**
 * 把 markdown 拆成 html / table 片段。
 * 只有「首行表头 + 第二行分隔行」的连续 `|...|` 块才认定为表格，
 * 否则原样交回 marked（避免误伤正文里以竖线开头的普通句子）。
 */
export function parseMarkdown(md: string): MdSegment[] {
  if (!md) return []

  const lines = md.split('\n')
  const segments: MdSegment[] = []
  let buffer: string[] = []

  const flushHtml = () => {
    const text = buffer.join('\n')
    buffer = []
    // 按空行切成独立段落块。
    // 关键：小程序 <rich-text> 支持的 CSS 是白名单制的，margin 不在其中，
    // 所以「段落间距」无法靠 rich-text 内部样式实现，必须在外面用原生 View 撑开。
    for (const block of text.split(/\n{2,}/)) {
      if (block.trim()) {
        segments.push({ kind: 'html', html: renderHtml(block) })
      }
    }
  }

  for (let i = 0; i < lines.length; i++) {
    if (!TABLE_LINE_RE.test(lines[i])) {
      buffer.push(lines[i])
      continue
    }

    // 收集连续的表格行
    const block: string[] = []
    while (i < lines.length && TABLE_LINE_RE.test(lines[i])) {
      block.push(lines[i])
      i++
    }
    i-- // 补偿，交给外层 for 的 i++

    const rows = block.map(splitRow)
    if (rows.length < 2 || !isSeparatorRow(rows[1])) {
      buffer.push(...block)
      continue
    }

    const headers = rows[0]
    const body = rows.slice(2).map((r) => {
      const cells = r.slice(0, headers.length)
      while (cells.length < headers.length) cells.push('')
      return cells
    })
    flushHtml()
    segments.push({ kind: 'table', table: { headers, rows: body } })
  }

  flushHtml()
  return segments
}

/** 保留：纯 HTML 出口（无表格场景或旧调用方） */
export function markdownToHtml(md: string): string {
  if (!md) return ''
  return renderHtml(md)
}
