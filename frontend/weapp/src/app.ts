import { PropsWithChildren } from 'react'
import { useLaunch } from '@tarojs/taro'
import './app.scss'

function App({ children }: PropsWithChildren<any>) {
  useLaunch(() => {
    // 启动钩子：后续可做 token 预检/静默登录占位。
  })
  return children
}

export default App
