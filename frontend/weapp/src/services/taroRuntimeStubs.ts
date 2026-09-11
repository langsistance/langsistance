/**
 * 仅供 `node --test` 使用的 Taro 运行时垫片。
 *
 * `@tarojs/taro` 会一路 import 到 `@tarojs/runtime`，而后者里
 * 有多处**裸标识符**常量（`if (ENABLE_INNER_HTML) { ... }`）。这些不是
 * 模块导出，是小程序构建期由 Taro 的 DefinePlugin 原地替换的编译期定义
 * （`@tarojs/webpack5-runner` 的 `MiniWebpackPlugin.getDefinePlugin()`）。
 * webpack 会做替换，node 不会——于是 `import '@tarojs/taro'` 在 node 里
 * 直接 `ReferenceError: ENABLE_INNER_HTML is not defined`。
 *
 * 本文件把构建期会 inlined 的那些常量按**同一个来源**补进全局：
 * 定义表是从 `MiniWebpackPlugin#getDefinePlugin()` 现取的，不手抄，
 * 免得 Taro 升级后这里悄悄过时。
 *
 * 只在测试进程里被 import；`src/services/resultsStore.ts` 不依赖它，
 * 所以小程序产物里不会出现这些全局量。
 */

import { createRequire } from 'node:module'

/** 与 `MiniWebpackPlugin.getDefinePlugin()` 的 runtimeConstants 同源取值。 */
type DefinePluginOptions = Record<string, string | boolean>

/**
 * 解析 `@tarojs/webpack5-runner` 的 CJS require。
 *
 * 用 `process.cwd()` 而不是 `import.meta.url`：本文件同时被 app 的
 * `tsc --noEmit`（module: commonjs）检查，那里不容许 `import.meta`。
 * 测试脚本总是在项目根跑（见 package.json 的 test），`process.cwd()`
 * 正是项目根，够用。
 */
const requireFromHere = createRequire(`${process.cwd()}/package.json`)

function readTaroDefineConstants(): DefinePluginOptions {
  try {
    const { MiniWebpackPlugin } = requireFromHere('@tarojs/webpack5-runner/dist/webpack/MiniWebpackPlugin')
    // 借原型拿到 getDefinePlugin，绕开构造函数对完整 combination 的依赖
    const probe = Object.create(MiniWebpackPlugin.prototype)
    probe.combination = { config: {}, buildAdapter: 'weapp' }
    return probe.getDefinePlugin().args[0] as DefinePluginOptions
  } catch {
    // Taro 内部结构变了就拿不到定义表——宁可直接失败，也不要塞一组
    // 可能已经不对的常量、让测试跑在一个假环境上。
    return {}
  }
}

/**
 * 把常量补进 `globalThis`，幂等。
 *
 * 只需在测试文件顶部 `import './taroRuntimeStubs.js'` 一次。
 */
export function installTaroRuntimeStubs(): void {
  const defineConstants = readTaroDefineConstants()
  if (Object.keys(defineConstants).length === 0) {
    throw new Error(
      'Taro 运行时常量表取不到（@tarojs/webpack5-runner 内部结构可能已变），' +
        '拒绝用假环境跑测试。请检查 MiniWebpackPlugin.getDefinePlugin()。',
    )
  }
  const target = globalThis as Record<string, unknown>
  for (const [name, value] of Object.entries(defineConstants)) {
    // `process.env.*` 形式的定义不需要（也不该）挂到 globalThis 上。
    if (name.startsWith('process.')) continue
    if (!(name in target)) target[name] = value
  }
}

installTaroRuntimeStubs()
