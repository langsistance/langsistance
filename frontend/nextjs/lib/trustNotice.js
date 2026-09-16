// 信息安全声明的文案唯一事实来源。
//
// 为什么不放进 lib/app-i18n/locales/*.ts：那两处是 TypeScript，tsconfig 的
// "moduleResolution": "bundler" 不允许 import 时带 .ts 扩展名，而 node --test
// 解析 ESM 相对导入必须带扩展名，两者冲突。纯 .js 模块同时满足 Next 打包与
// 内置测试运行器。
//
// 同步约束：TRUST_NOTES 是 TRUST_COPY.items 在界面空间受限处的缩写子集，
// 不是独立文案。改动 items 时必须回头核对 TRUST_NOTES 两句是否仍然成立。
// lib/trustNotice.test.mjs 会校验结构一致，但校验不了语义漂移——靠人看。

export const TRUST_COPY = {
  zh: {
    headline: '您的专有信息只留在您的账号里',
    items: [
      { key: 'train', label: '不外传', desc: '提问与分析结果不用于训练任何 AI 模型' },
      { key: 'isolate', label: '不外泄', desc: '对话记录按账号隔离，仅您本人可读取' },
      { key: 'upload', label: '不外流', desc: '上传的专利文件处理完成后从服务器删除' },
    ],
    more: '了解更多',
  },
  en: {
    headline: 'Your proprietary information stays in your account',
    items: [
      { key: 'train', label: 'Not shared', desc: 'Prompts and results are never used to train AI models' },
      { key: 'isolate', label: 'Not exposed', desc: 'Conversations are isolated by account and readable only by you' },
      { key: 'upload', label: 'Not retained', desc: 'Uploaded patent files are deleted from our servers after processing' },
    ],
    more: 'Learn more',
  },
}

export const TRUST_NOTES = {
  zh: {
    login: '🔒 传输全程加密 · 您的内容不会用于训练 AI 模型',
    focus: '您的内容不会用于训练 AI 模型，上传文件处理完成后删除',
  },
  en: {
    login: '🔒 Encrypted in transit · Your content is never used to train AI models',
    focus: 'Your content is never used to train AI models; uploads are deleted after processing',
  },
}
