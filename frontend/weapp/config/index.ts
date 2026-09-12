// Taro 构建配置（weapp 目标）。API 基址单一出口:
// 编译期 env TARO_APP_API_BASE 覆盖, 未设则回退本地联调地址
// (开发者工具勾选"不校验合法域名"后可直连; 上线前替换为正式网关地址)。
const apiBase = process.env.TARO_APP_API_BASE || 'http://127.0.0.1:7777'

const config = {
  projectName: 'copiioai-weapp',
  date: '2026-09-09',
  designWidth: 750,
  deviceRatio: {
    640: 2.34 / 2,
    750: 1,
    375: 2,
    828: 1.81 / 2,
  },
  sourceRoot: 'src',
  outputRoot: 'dist',
  plugins: [],
  defineConstants: {
    'process.env.TARO_APP_API_BASE': JSON.stringify(apiBase),
  },
  copy: {
    patterns: [],
    options: {},
  },
  framework: 'react',
  compiler: {
    type: 'webpack5',
    prebundle: { enable: false },
  },
  cache: {
    enable: false,
  },
  mini: {
    // Taro 默认只把 taro 自家的 node_modules 交 babel（MiniWebpackModule.js:164-167），
    // 其余依赖原样放行。marked 发的是 ES2022（裸类字段 `options;` / `#私有方法`），
    // 开发者工具桌面引擎能跑，但**真机调试**的解析器直接报
    // "SyntaxError: Unexpected token ;"（pages/chat/index.js, 1:19377）。
    // 这里把它拉进转译范围，只降语法、不改行为。谓词写法对齐 Taro 自己的默认项。
    compile: {
      include: [
        (filename: string) => /(?<=node_modules[\\/])marked/.test(filename),
      ],
    },
    postcss: {
      pxtransform: {
        enable: true,
        config: {},
      },
      cssModules: {
        enable: false,
      },
    },
  },
  h5: {
    publicPath: '/',
    staticDirectory: 'static',
    postcss: {
      autoprefixer: {
        enable: true,
        config: {},
      },
      cssModules: {
        enable: false,
      },
    },
  },
}

module.exports = function (merge) {
  if (process.env.NODE_ENV === 'development') {
    return merge({}, config, require('./dev'))
  }
  return merge({}, config, require('./prod'))
}
