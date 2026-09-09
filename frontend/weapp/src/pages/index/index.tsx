import { useCallback, useState } from 'react'
import { Button, Text, View } from '@tarojs/components'
import Taro, { useDidShow } from '@tarojs/taro'
import { isLoggedIn } from '../../services/auth'
import { fetchSessions, SessionItem } from '../../services/sessions'
import { errorText } from '../../services/api'
import './index.scss'

/** 会话列表（DeepSeek-app 式首页）：未登录 → 空态引导登录。 */
export default function IndexPage() {
  const [sessions, setSessions] = useState<SessionItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = useCallback(async () => {
    if (!isLoggedIn()) return
    setLoading(true)
    setError('')
    try {
      setSessions(await fetchSessions())
    } catch (err) {
      setError(errorText(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useDidShow(() => {
    if (!isLoggedIn()) {
      Taro.navigateTo({ url: '/pages/login/index' })
      return
    }
    load()
  })

  function newChat() {
    Taro.navigateTo({ url: '/pages/chat/index' })
  }

  function openSession(session: SessionItem) {
    Taro.navigateTo({
      url: `/pages/chat/index?session_id=${session.session_id}`,
    })
  }

  function goLogin() {
    Taro.navigateTo({ url: '/pages/login/index' })
  }

  return (
    <View className='sessions'>
      <View className='sessions-head'>
        <Text className='sessions-title'>对话</Text>
        <View className='sessions-new' onClick={newChat}>
          <Text className='sessions-new-plus'>＋</Text>
        </View>
      </View>

      {!isLoggedIn() ? (
        <View className='sessions-empty'>
          <Text className='sessions-empty-text text-muted'>登录后查看历史对话</Text>
          <Button className='btn-primary sessions-empty-btn' onClick={goLogin}>
            微信登录
          </Button>
        </View>
      ) : (
        <View className='sessions-list'>
          {sessions.map((s) => (
            <View
              key={s.session_id}
              className='card sessions-item'
              onClick={() => openSession(s)}
            >
              <Text className='sessions-item-title'>
                {s.title || '未命名对话'}
              </Text>
              <Text className='sessions-item-time text-muted'>
                {s.update_time ? s.update_time.replace('T', ' ').slice(0, 16) : ''}
              </Text>
            </View>
          ))}
          {!loading && !error && sessions.length === 0 ? (
            <View className='sessions-empty'>
              <Text className='sessions-empty-text text-muted'>
                还没有对话，点击右下角开始
              </Text>
            </View>
          ) : null}
          {error ? <Text className='sessions-error'>{error}</Text> : null}
        </View>
      )}
    </View>
  )
}
