import { Text, View } from '@tarojs/components'
import { useRouter } from '@tarojs/taro'
import './index.scss'

/**
 * M2 对话页（当前为占位骨架）：
 * - 流式对话渲染（SSE / WebSocket 桥，M2 接入）
 * - 专利结果面板（复用后端检索/结果展示架构）
 */
export default function ChatPage() {
  const router = useRouter()
  const sessionId = router.params.session_id || ''

  return (
    <View className='chat'>
      <View className='chat-empty'>
        <Text className='chat-empty-title'>专利智能对话</Text>
        <Text className='chat-empty-sub text-muted'>
          {sessionId ? `会话 ${sessionId.slice(0, 8)}…（M2 开发中）` : '正在开发对话能力（M2）'}
        </Text>
      </View>
    </View>
  )
}
