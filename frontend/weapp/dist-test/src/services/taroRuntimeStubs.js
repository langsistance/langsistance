"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.installTaroRuntimeStubs = installTaroRuntimeStubs;
const node_module_1 = require("node:module");
/**
 * 本模块所在目录，跨 CJS/ESM 两种形态都成立。
 *
 * 定义期存在 `__dirname`（CJS 全局）；本文件**同时**被测试侧按 ESM 加载，
 * 那里没有它——但 ESM 下 `import.meta.url` 有值。两边各取其一。
 *
 * 不能用 `new URL('./package.json', import.meta.url)` 当作唯一写法：
 * `tsc --noEmit`（module: commonjs）不容许 `import.meta`，会直接编译失败，
 * 所以只能在 `typeof __dirname` 判空之后、由 `eval` 间接取一次。
 */
function selfDir() {
    const cjsDir = typeof __dirname === 'string' ? __dirname : '';
    if (cjsDir)
        return cjsDir;
    const importMetaUrl = (0, eval)('import.meta.url');
    // 只用 URL 解析路径，不 fs 读文件——故不依赖本模块在磁盘上的存在形式。
    return decodeURIComponent(new URL('.', importMetaUrl).pathname.replace(/^\/(?=[A-Za-z]:)/, ''));
}
/** 向上逐级找 package.json，得到锚在本模块而非 cwd 的 require。 */
function makeRequireFromSelf() {
    let dir = selfDir();
    for (;;) {
        try {
            return (0, node_module_1.createRequire)(`${dir}/package.json`);
        }
        catch {
            const parent = dir.replace(/[\\/][^\\/]*$/, '');
            if (!parent || parent === dir)
                throw new Error(`taroRuntimeStubs: 自 ${selfDir()} 起未找到 package.json`);
            dir = parent;
        }
    }
}
const requireFromHere = makeRequireFromSelf();
/**
 * `@tarojs/webpack5-runner` 的解析锚点（其 package.json 所在目录）。
 *
 * 取不到定义表时要把这个路径写进报错——否则在多包/workspace 场景下
 * 只能看到一句"Taro 内部结构可能已变"，而真实原因往往是**找错了目录**。
 */
const TARO_RUNNER_SPECIFIER = '@tarojs/webpack5-runner/dist/webpack/MiniWebpackPlugin';
function taroRunnerAnchor() {
    try {
        return requireFromHere.resolve(TARO_RUNNER_SPECIFIER);
    }
    catch {
        return `(未解析到 ${TARO_RUNNER_SPECIFIER}；require 锚点为 ${selfDir()})`;
    }
}
function readTaroDefineConstants() {
    try {
        const { MiniWebpackPlugin } = requireFromHere(TARO_RUNNER_SPECIFIER);
        // 借原型拿到 getDefinePlugin，绕开构造函数对完整 combination 的依赖
        const probe = Object.create(MiniWebpackPlugin.prototype);
        probe.combination = { config: {}, buildAdapter: 'weapp' };
        return probe.getDefinePlugin().args[0];
    }
    catch {
        // Taro 内部结构变了就拿不到定义表——宁可直接失败，也不要塞一组
        // 可能已经不对的常量、让测试跑在一个假环境上。
        return {};
    }
}
/**
 * 把常量补进 `globalThis`，幂等。
 *
 * 只需在测试文件顶部 `import './taroRuntimeStubs.js'` 一次。
 */
function installTaroRuntimeStubs() {
    const defineConstants = readTaroDefineConstants();
    if (Object.keys(defineConstants).length === 0) {
        throw new Error('Taro 运行时常量表取不到，拒绝用假环境跑测试。' +
            `已尝试从 ${selfDir()} 解析 ${TARO_RUNNER_SPECIFIER}，解析结果为：${taroRunnerAnchor()}。` +
            '若该路径不在本项目 node_modules 内，多半是解析锚点落在了错误的目录；' +
            '若路径正确，再检查 MiniWebpackPlugin.getDefinePlugin() 是否已随 Taro 升级改名。');
    }
    const target = globalThis;
    for (const [name, value] of Object.entries(defineConstants)) {
        // `process.env.*` 形式的定义不需要（也不该）挂到 globalThis 上。
        if (name.startsWith('process.'))
            continue;
        if (!(name in target))
            target[name] = value;
    }
}
installTaroRuntimeStubs();
//# sourceMappingURL=taroRuntimeStubs.js.map