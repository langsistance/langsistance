import { Image, Text, View } from '@tarojs/components'
// 品牌图：由 E:/online/酷彼智能/copiioai/jx_logo.png 处理而来——去掉四周 1px
// 导出描边、裁到内容、白底抠成透明、缩到 480px 宽并存为 128 色调色板 PNG（23KB）。
// 原图 1254×1254/868KB，且是 RGB 无透明通道，直接用会在 #f6f8fa 底色上露出白方块。
import brandLogo from '../../assets/brand-logo.png'
import './index.scss'

/**
 * 品牌区：品牌图 + 口号。
 *
 * 登录页与新对话空态**共用同一份内容**——抽成组件而不是两处各写一遍，
 * 是为了改文案/换图时不会只改一处、两边慢慢跑偏。
 * 各自需要额外的说明文字（如新对话页的副标题与示例）由调用方在下方自行补。
 */
export default function BrandBlock() {
  return (
    <View className='brand'>
      <Image className='brand-logo' src={brandLogo} mode='widthFix' />
      <Text className='brand-title'>专利情报，一问即得</Text>
    </View>
  )
}
