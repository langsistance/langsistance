import { Text, View } from '@tarojs/components'
import { PickedFile, extensionOf } from '../../services/upload'
import './index.scss'

type Props = {
  file: PickedFile | null
  onRemove: () => void
}

/** 对齐 web 的 getFileTypeBadge（ChatComposer.tsx:80-85）。 */
export function fileBadge(name: string): string {
  const ext = extensionOf(name)
  if (ext === '.docx') return 'DOCX'
  if (ext === '.xml') return 'XML'
  return 'PDF'
}

/** 对齐 web 的 formatFileSize（ChatComposer.tsx:87-91）。 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/**
 * 已选文件的 chip 条，挂在输入框**上方**，只在有文件时渲染——没文件时
 * 整条不出现（此前这里还兼着「＋」按钮，于是空着时也留一条空横条）。
 * 「＋」按钮现在在输入框左侧同一行，见 pages/chat 的 .chat-attach。
 */
export default function AttachmentBar({ file, onRemove }: Props) {
  if (!file) return null
  return (
    <View className='attach-bar'>
      <View className='attach-chip'>
        <Text className='attach-chip-badge'>{fileBadge(file.name)}</Text>
        <Text className='attach-chip-name'>{file.name}</Text>
        <Text className='attach-chip-size'>{formatFileSize(file.size)}</Text>
        <Text className='attach-chip-remove' onClick={onRemove}>
          ✕
        </Text>
      </View>
    </View>
  )
}
