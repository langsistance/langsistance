import { useMemo } from 'react'
import { View, Text } from '@tarojs/components'
import Taro from '@tarojs/taro'
import './index.scss'

type Props = {
  title: string
  onMenuClick: () => void
  onNewChat: () => void
}

const FALLBACK_NAV_HEIGHT = 44
/** 取不到胶囊时的兜底占宽（px）：微信胶囊宽约 87px + 距右约 7px */
const FALLBACK_CAPSULE_WIDTH = 94
/**
 * 顶部左侧「☰ ＋」整组的宽度，单位 rpx，必须与 index.scss 一致：
 *   ☰ 的 margin-left(12) + ☰ 宽(72) + ＋ 的 margin-left(4) + ＋ 宽(72) = 160
 */
const LEFT_GROUP_RPX = 160
/** 标题与左右内容之间的间隙（px） */
const TITLE_GAP = 12

/**
 * 自绘顶栏（页面配 navigationStyle: 'custom' 后系统导航栏不再渲染）。
 *
 * 高度按微信官方公式算：
 *   navBarHeight = (胶囊top - 状态栏高度) * 2 + 胶囊高度
 *
 * 水平方向是这道题真正的难点：右上角不是空地，是**系统胶囊**（固定约 94px，
 * 且盖在自绘层之上，压缩不了）。
 *
 *   375px 屏：  ├─ ☰ ─┤├─ ＋ ─┤            ├── 胶囊 ──┤
 *              0      100                   281      375
 *
 * 于是「标题屏幕居中」要求左右内缩**相等**，而右侧还得让开胶囊 —— 若照搬
 * 常见做法（☰ 独占左侧 50px、＋ 贴胶囊左侧），左 50 / 右 146 不对称，标题会
 * 明显偏左。所以改为把 ☰ 与 ＋ **并排在左侧**成一组，标题两侧同取
 *   inset = max(左侧组宽, 胶囊占宽) + 间隙
 * 两边同值 ⇒ 中心恒等于屏幕中心；取 max 保证窄屏上右侧也不会钻到胶囊下面。
 *
 * 注意：本组件的内联样式走 px，**不经过 Taro 的 pxtransform**（那只处理 .scss），
 * 而胶囊 API 返回的也是 px，两边同单位，故这里一律用 px 而非 rpx。
 */
export default function NavBar({ title, onMenuClick, onNewChat }: Props) {
  const { statusBarHeight, navBarHeight, titleInset } = useMemo(() => {
    const info = Taro.getSystemInfoSync()
    const status = info.statusBarHeight || 0
    const windowWidth = info.windowWidth || 375
    let height = FALLBACK_NAV_HEIGHT
    let capsuleWidth = FALLBACK_CAPSULE_WIDTH
    try {
      const menu = Taro.getMenuButtonBoundingClientRect()
      if (menu && menu.height) {
        height = (menu.top - status) * 2 + menu.height
      }
      if (menu && menu.left > 0 && windowWidth > menu.left) {
        // 胶囊左缘到屏幕右边的距离
        capsuleWidth = windowWidth - menu.left
      }
    } catch {
      // 取不到胶囊（非微信端/调试环境）时用兜底值
    }
    const leftGroupWidth = (LEFT_GROUP_RPX * windowWidth) / 750
    const inset = Math.max(leftGroupWidth, capsuleWidth) + TITLE_GAP
    return {
      statusBarHeight: status,
      navBarHeight: height,
      titleInset: inset,
    }
  }, [])

  return (
    <View className='navbar' style={{ paddingTop: `${statusBarHeight}px` }}>
      <View className='navbar-inner' style={{ height: `${navBarHeight}px` }}>
        <View className='navbar-menu' onClick={onMenuClick}>
          {/* 菜单图标也是画的：三条等距圆角杠 */}
          <View className='navbar-menu-icon'>
            <View className='navbar-menu-bar navbar-menu-bar-top' />
            <View className='navbar-menu-bar navbar-menu-bar-middle' />
            <View className='navbar-menu-bar navbar-menu-bar-bottom' />
          </View>
        </View>
        <View className='navbar-new' onClick={onNewChat}>
          {/* 「＋」是画出来的（两根圆角杆），不是字体字形——理由见 index.scss */}
          <View className='navbar-new-icon'>
            <View className='navbar-new-bar navbar-new-bar-h' />
            <View className='navbar-new-bar navbar-new-bar-v' />
          </View>
        </View>
        {/* 左右同值内缩 ⇒ 中心 = 屏幕中心；取到的值已按胶囊/左侧组宽较大者算过 */}
        <Text
          className='navbar-title'
          style={{ left: `${titleInset}px`, right: `${titleInset}px` }}
        >
          {title}
        </Text>
      </View>
    </View>
  )
}
