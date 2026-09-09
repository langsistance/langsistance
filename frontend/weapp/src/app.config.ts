// Taro 4.1 的 defineAppConfig 经全局类型注入；此处用纯对象导出等效
// （taro build 按约定读取默认导出对象作为 app 配置）。
const config = {
  pages: [
    'pages/index/index', // 会话列表（首页）
    'pages/chat/index',  // 对话页（M2）
    'pages/login/index', // 微信登录（M1）
  ],
  window: {
    backgroundTextStyle: 'light',
    navigationBarBackgroundColor: '#f6f8fa',
    navigationBarTitleText: 'CopiioAI 专利助手',
    navigationBarTextStyle: 'black',
    backgroundColor: '#f6f8fa',
  },
  lazyCodeLoading: 'requiredComponents',
}

export default config

