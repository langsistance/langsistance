import { useEffect, useState } from 'react'
import { Button, Input, Text, View } from '@tarojs/components'
import './index.scss'

type Props = {
  visible: boolean
  initialTitle: string
  busy?: boolean
  error?: string
  onCancel: () => void
  onConfirm: (title: string) => void
}

const MAX_TITLE_LEN = 60

/**
 * 重命名弹窗（纯展示，不碰网络）。
 * 自绘理由：微信小程序没有带输入框的原生弹窗 —— Taro.showModal 只有
 * 确定/取消，window.prompt 在小程序不存在。
 */
export default function RenameModal({
  visible,
  initialTitle,
  busy,
  error,
  onCancel,
  onConfirm,
}: Props) {
  const [value, setValue] = useState(initialTitle)

  // 每次打开都用当前标题重新预填
  useEffect(() => {
    if (visible) setValue(initialTitle)
  }, [visible, initialTitle])

  if (!visible) return null

  const trimmed = value.trim()
  const canSubmit = trimmed.length > 0 && !busy

  return (
    <View className='rename-mask' catchMove>
      <View className='rename-box'>
        <Text className='rename-heading'>重命名对话</Text>
        <Input
          className='rename-input'
          value={value}
          maxlength={MAX_TITLE_LEN}
          focus
          placeholder='输入新的标题'
          placeholderClass='rename-placeholder'
          onInput={(e) => setValue(e.detail.value)}
        />
        {error ? <Text className='rename-err'>{error}</Text> : null}
        <View className='rename-actions'>
          <Button className='rename-btn' disabled={busy} onClick={onCancel}>
            取消
          </Button>
          <Button
            className='rename-btn rename-btn-primary'
            disabled={!canSubmit}
            onClick={() => onConfirm(trimmed)}
          >
            {busy ? '保存中…' : '保存'}
          </Button>
        </View>
      </View>
    </View>
  )
}
