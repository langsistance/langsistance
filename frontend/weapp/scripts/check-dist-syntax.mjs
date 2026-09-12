/**
 * 产物语法闸门：dist 里每个 .js 都必须能被按 ES2015 解析。
 *
 * 缘起（2026-09-12）：marked@18 发的是 ES2022（裸类字段 `options;`、`#私有方法`），
 * 而 Taro 默认只把 **taro 自家的** node_modules 交给 babel
 * （`@tarojs/webpack5-runner/dist/webpack/MiniWebpackModule.js:164-167`），
 * 其余依赖原样进产物。开发者工具的桌面引擎能跑，**真机调试**的解析器直接报
 *   SyntaxError: Unexpected token ;   (pages/chat/index.js, 1:19377)
 * 这个缺陷在工具里完全看不出来，只在真机上炸 —— 所以闸门放在构建之后自动跑。
 *
 * 修的是 config/index.ts 的 `mini.compile.include`；本脚本是它的回归保险。
 *
 * 为什么用 ecmaVersion 2015：package.json 的 browserslist 是 `ios >= 8` /
 * `Android >= 4.1`，比 2015 更保守。实测修复后的产物在 2015 档全量可解析，
 * 用它当闸门既精确（解析器判定，不会有正则那种把 CSS 颜色 `#e1e4e8;`
 * 误判成私有字段的假阳性）又留有充足余量。
 *
 * 注：acorn 是 webpack@5.91.0 的传递依赖，非本项目直接声明。
 */
import fs from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const acorn = require('acorn')

const ES_TARGET = 2015
const DIST = 'dist'

if (!fs.existsSync(DIST)) {
  // 不静默通过：本项目吃过"空匹配报绿"的亏（node --test 的 globstar 曾扫空仍 pass）
  console.error(`✗ 找不到 ${DIST}/ —— 先跑 npm run build:weapp 再检查`)
  process.exit(1)
}

const files = []
;(function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name)
    if (entry.isDirectory()) walk(p)
    else if (entry.name.endsWith('.js')) files.push(p)
  }
})(DIST)

if (files.length === 0) {
  console.error(`✗ ${DIST}/ 下一个 .js 都没有 —— 构建产物不完整`)
  process.exit(1)
}

const failures = []
for (const file of files) {
  const source = fs.readFileSync(file, 'utf8')
  try {
    acorn.parse(source, { ecmaVersion: ES_TARGET })
  } catch (err) {
    const rel = file.split(path.sep).join('/')
    failures.push(`${rel}:${err.loc?.line ?? '?'}:${err.loc?.column ?? '?'}  ${err.message}`)
  }
}

if (failures.length > 0) {
  console.error(`✗ ${failures.length}/${files.length} 个产物含 ES${ES_TARGET} 之后的语法，真机会解析失败：`)
  for (const f of failures) console.error('   ' + f)
  console.error('')
  console.error('  多半是某个 node_modules 没被转译。把它们加进 config/index.ts 的')
  console.error('  mini.compile.include（谓词写法见该文件的 marked 项）。')
  process.exit(1)
}

console.log(`✔ ${files.length} 个产物 js 全部可按 ES${ES_TARGET} 解析（真机可解析）`)
