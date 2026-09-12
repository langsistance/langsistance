/**
 * 静态资源模块声明。
 *
 * Taro 的 webpack 配置用 url-loader 处理图片：小于阈值的内联成 data URI，
 * 超过阈值的落到 dist 并在 import 处返回路径字符串。两种情况下 import 的
 * 结果都是 string，但 TypeScript 不知道 `.png` 是模块，会报
 * "Cannot find module ... or its corresponding type declarations"。
 *
 * tsconfig.json 的 include 里已经列了 `./types`，放这里即可自动生效。
 */
declare module '*.png' {
  const src: string
  export default src
}

declare module '*.jpg' {
  const src: string
  export default src
}

declare module '*.jpeg' {
  const src: string
  export default src
}

declare module '*.svg' {
  const src: string
  export default src
}
