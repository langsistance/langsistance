import { PropsWithChildren } from 'react'
import { useLaunch } from '@tarojs/taro'
import { registerPrivacyHandler } from './services/privacy'
import './app.scss'

function App({ children }: PropsWithChildren<any>) {
  useLaunch(() => {
    // 隐私授权监听必须在 App 级注册一次——页面级注册会在切页时丢失
    // pendingResolve，导致用户点同意后 Promise 永远不兑现。
    registerPrivacyHandler()
  })
  return children
}

export default App
