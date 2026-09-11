import { useState } from 'react'
import { Button, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import { wechatLogin } from '../../services/auth'
import { errorText } from '../../services/api'
import './index.scss'

/**
 * M1 登录页：微信一键登录。
 * 成功后回首页（会话列表）；AppID 未就绪时给出可读错误。
 */
export default function LoginPage() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function onLogin() {
    if (loading) return
    setLoading(true)
    setError('')
    try {
      await wechatLogin()
      Taro.navigateBack({
        fail: () => Taro.reLaunch({ url: '/pages/chat/index' }),
      })
    } catch (err) {
      setError(errorText(err, '登录失败，请稍后重试'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <View className='login'>
      <View className='login-brand'>
        <View className='login-logo' />
        <Text className='login-title'>CopiioAI 专利助手</Text>
        <Text className='login-sub text-muted'>
          专利检索 · 侵权风险 · 授权前景分析
        </Text>
      </View>

      <Button
        className='btn-primary login-btn'
        loading={loading}
        disabled={loading}
        onClick={onLogin}
      >
        微信一键登录
      </Button>

      {error ? <Text className='login-error'>{error}</Text> : null}
      <Text className='login-legal text-muted'>
        登录即代表同意《用户协议》与《隐私政策》
      </Text>
    </View>
  )
}
