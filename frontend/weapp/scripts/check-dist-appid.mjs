/**
 * 构建后闸门：确认 dist/project.config.json 的 appid 就是源头那个。
 *
 * 缘起（2026-09-12）：开发者工具导入的是 **frontend/weapp/dist**，它把项目设置
 * 写进 `dist/project.config.json`（appid 改成当前在用的那个）；而每次 `taro build`
 * 都会把源头 `frontend/weapp/project.config.json` 拷进 dist 覆盖掉它。
 *
 * 症状极具误导性：工具里看着 appid 是对的，构建一次就变回旧值，而**根本看不出
 * 是谁改的**。当时为此排查了很久，真因是源头那份 appid 一直是旧的测试号，
 * 于是每次构建都把工具里改好的值打回去 —— 继而使 wx.login 的 code 用错 appid
 * 签发，服务端兑换时报 `40029 invalid code`，表现为「微信一键登录一直失败」。
 *
 * 所以这里只守一条不变量：**dist 的 appid 必须等于源头的 appid**。
 * 一旦 Taro 的拷贝链路变化（或有人绕过 npm script 构建），这里会立刻报出来，
 * 而不是等到某天登录又挂了才发现。
 *
 * 注：miniprogramRoot 不需要管 —— 实测 Taro 会自己把它规范成相对输出根的值
 * （源头写 "dist/"，dist 里写 "./"），两份文件在该字段上本就应当不同。
 */
import fs from 'node:fs'

const DIST_CONFIG = 'dist/project.config.json'
const SOURCE_CONFIG = 'project.config.json'

function readAppid(path) {
  if (!fs.existsSync(path)) return { missing: true }
  try {
    return { appid: JSON.parse(fs.readFileSync(path, 'utf8')).appid }
  } catch (err) {
    return { error: err.message }
  }
}

const dist = readAppid(DIST_CONFIG)
const source = readAppid(SOURCE_CONFIG)

if (dist.missing) {
  console.error(`✗ 找不到 ${DIST_CONFIG} —— 先跑 taro build`)
  process.exit(1)
}
if (dist.error) {
  console.error(`✗ ${DIST_CONFIG} 不是合法 JSON：${dist.error}`)
  process.exit(1)
}
if (source.missing || source.error) {
  console.error(`✗ 读不到源头 ${SOURCE_CONFIG} 的 appid`)
  process.exit(1)
}

if (dist.appid !== source.appid) {
  console.error(
    `✗ appid 不一致：dist=${dist.appid}  源头=${source.appid}\n` +
      '  构建拷贝链路可能已变。工具里看到的 appid 与真正构建进去的不是同一个，\n' +
      '  典型后果是 wx.login 的 code 用错 appid 签发、服务端报 40029。',
  )
  process.exit(1)
}

console.log(`✔ dist 与源头的 appid 一致：${dist.appid}`)
