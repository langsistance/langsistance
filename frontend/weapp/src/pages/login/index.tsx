import { useState } from 'react'
import { Button, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import { wechatLogin } from '../../services/auth'
import { errorText } from '../../services/api'
import BrandBlock from '../../components/BrandBlock'
import './index.scss'

/**
 * M1 登录页：微信一键登录。
 * 成功后回首页（对话页）；AppID 未就绪时给出可读错误。
 */
export default function LoginPage() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  /** 协议勾选。未勾选不放行登录——这是显式同意，不是「登录即代表同意」 */
  const [agreed, setAgreed] = useState(false)

  async function onLogin() {
    if (loading || !agreed) return
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

  function openPrivacy() {
    Taro.navigateTo({
      url: '/pages/privacy/index',
      fail: () => Taro.redirectTo({ url: '/pages/privacy/index' }),
    })
  }

  return (
    <View className='login'>
      <View className='login-brand'>
        <BrandBlock />
      </View>

      <Button
        className='btn-primary login-btn'
        loading={loading}
        disabled={loading || !agreed}
        onClick={onLogin}
      >
        微信一键登录
      </Button>

      {error ? <Text className='login-error'>{error}</Text> : null}

      {/* 整行可点切换勾选；点《隐私政策》时 stopPropagation，避免顺带切换 */}
      <View className='login-agree' onClick={() => setAgreed((v) => !v)}>
        <View className={`login-check${agreed ? ' login-check-on' : ''}`}>
          {agreed ? <Text className='login-check-tick'>✓</Text> : null}
        </View>
        <Text className='login-agree-text text-muted'>
          我已阅读并同意
          <Text
            className='login-legal-link'
            onClick={(e) => {
              e.stopPropagation()
              openPrivacy()
            }}
          >
            《隐私政策》
          </Text>
        </Text>
      </View>
    </View>
  )
}
