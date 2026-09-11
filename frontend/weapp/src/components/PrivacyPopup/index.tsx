import { Button, Text, View } from '@tarojs/components'
// 只引这一个常量：同意按钮 id 与 resolvePrivacy 的默认 id 一旦漂移，
// 微信会判定同意无效（核验 buttonId 确实被点击过），且静默失败。
import { PRIVACY_AGREE_BUTTON_ID } from '../../services/privacy'
import './index.scss'

type Props = {
  visible: boolean
  onAgree: () => void
  onDecline: () => void
}

/**
 * 隐私授权浮层。必须用 <button open-type="agreePrivacyAuthorization">
 * 收集用户同意——普通 onClick 不算数，微信不认。
 */
export default function PrivacyPopup({ visible, onAgree, onDecline }: Props) {
  if (!visible) return null
  return (
    <View className='privacy-mask'>
      <View className='privacy-panel'>
        <Text className='privacy-title'>隐私保护指引</Text>
        <Text className='privacy-body'>
          为完成专利文档分析，我们需要读取你在微信会话中选择的文件。
          文件仅用于本次分析，不会用于其他用途。
          请阅读并同意《隐私保护指引》后继续。
        </Text>
        <View className='privacy-actions'>
          <Button className='privacy-btn privacy-btn-ghost' onClick={onDecline}>
            暂不同意
          </Button>
          <Button
            id={PRIVACY_AGREE_BUTTON_ID}
            className='privacy-btn privacy-btn-primary'
            openType='agreePrivacyAuthorization'
            onAgreePrivacyAuthorization={onAgree}
          >
            同意并继续
          </Button>
        </View>
      </View>
    </View>
  )
}
