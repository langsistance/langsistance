// Taro 4.1 的 defineAppConfig 经全局类型注入；此处用纯对象导出等效
// （taro build 按约定读取默认导出对象作为 app 配置）。
const config = {
  pages: [
    'pages/chat/index',    // 对话页（形态一：首页即对话页）
    'pages/login/index',   // 微信登录（M1）
    'pages/privacy/index', // 隐私政策（M3）
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

