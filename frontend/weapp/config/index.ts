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
