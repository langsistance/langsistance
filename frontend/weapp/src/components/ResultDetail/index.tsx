import { Text, View } from '@tarojs/components'
import { ResultsPayload } from '../../utils/results'
import './index.scss'

type Props = {
  payload: ResultsPayload
  row: Record<string, string>
  visible: boolean
  onClose: () => void
}

/** Task 6 会把它换成真正的三 tab 详情。 */
export default function ResultDetail({ visible, onClose }: Props) {
  if (!visible) return null
  return (
    <View className='rd' onClick={onClose}>
      <Text className='rd-placeholder'>详情面板（Task 6 实现）</Text>
    </View>
  )
}
