import { ScrollView, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import type { SessionItem } from '../../services/sessions'
import './index.scss'

type Props = {
  visible: boolean
  sessions: SessionItem[]
  currentSessionId: string
  loading?: boolean
  error?: string
  onClose: () => void
  onSelect: (sessionId: string) => void
  onNew: () => void
  onRename: (session: SessionItem) => void
  onDelete: (session: SessionItem) => void
}

const ACTIONS = ['重命名', '删除']

/** 会话抽屉（纯展示：只发意图，不碰网络）。 */
export default function SessionDrawer({
  visible,
  sessions,
  currentSessionId,
  loading,
  error,
  onClose,
  onSelect,
  onNew,
  onRename,
  onDelete,
}: Props) {
  function showActions(session: SessionItem) {
    Taro.showActionSheet({
      itemList: ACTIONS,
      success: (res) => {
        if (res.tapIndex === 0) onRename(session)
        else if (res.tapIndex === 1) onDelete(session)
      },
      fail: () => {},
    })
  }

  if (!visible) return null

  return (
    <View className='drawer'>
      <View className='drawer-mask' onClick={onClose} catchMove />
      <View className='drawer-panel'>
        <View className='drawer-new' onClick={onNew}>
          <Text className='drawer-new-plus'>＋</Text>
          <Text className='drawer-new-text'>新对话</Text>
        </View>

        <ScrollView className='drawer-scroll' scrollY>
          {sessions.map((s) => (
            <View
              key={s.session_id}
              className={`drawer-item${
                s.session_id === currentSessionId ? ' drawer-item-active' : ''
              }`}
              onClick={() => onSelect(s.session_id)}
            >
              <Text className='drawer-item-title'>
                {s.title || '未命名对话'}
              </Text>
              <View
                className='drawer-item-more'
                onClick={(e) => {
                  e.stopPropagation()
                  showActions(s)
                }}
              >
                <Text className='drawer-item-more-icon'>…</Text>
              </View>
            </View>
          ))}

          {!loading && !error && sessions.length === 0 ? (
            <View className='drawer-empty'>
              <Text className='text-muted'>还没有对话</Text>
            </View>
          ) : null}
          {loading ? (
            <View className='drawer-empty'>
              <Text className='text-muted'>加载中…</Text>
            </View>
          ) : null}
          {error ? (
            <View className='drawer-empty'>
              <Text className='drawer-error'>{error}</Text>
            </View>
          ) : null}
        </ScrollView>
      </View>
    </View>
  )
}
