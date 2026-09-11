import { Text, View } from '@tarojs/components'
import { PickedFile, extensionOf } from '../../services/upload'
import './index.scss'

type Props = {
  file: PickedFile | null
  busy?: boolean
  onAdd: () => void
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

export default function AttachmentBar({ file, busy, onAdd, onRemove }: Props) {
  return (
    <View className='attach-bar'>
      <View
        className={`attach-btn${busy ? ' attach-btn-busy' : ''}`}
        onClick={busy ? undefined : onAdd}
      >
        <Text className='attach-btn-icon'>＋</Text>
      </View>
      {file ? (
        <View className='attach-chip'>
          <Text className='attach-chip-badge'>{fileBadge(file.name)}</Text>
          <Text className='attach-chip-name'>{file.name}</Text>
          <Text className='attach-chip-size'>{formatFileSize(file.size)}</Text>
          <Text className='attach-chip-remove' onClick={onRemove}>
            ✕
          </Text>
        </View>
      ) : null}
    </View>
  )
}
