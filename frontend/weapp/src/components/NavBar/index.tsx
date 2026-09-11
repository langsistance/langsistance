import { useMemo } from 'react'
import { Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import './index.scss'

type Props = {
  title: string
  onMenuClick: () => void
}

const FALLBACK_NAV_HEIGHT = 44

/**
 * 自绘顶栏（页面配 navigationStyle: 'custom' 后系统导航栏不再渲染）。
 * 高度按微信官方公式算，避免与右上角胶囊按钮错位：
 *   navBarHeight = (胶囊top - 状态栏高度) * 2 + 胶囊高度
 * 胶囊在右上角、☰ 在左上角，水平方向不冲突，只需对齐垂直。
 */
export default function NavBar({ title, onMenuClick }: Props) {
  const { statusBarHeight, navBarHeight } = useMemo(() => {
    const info = Taro.getSystemInfoSync()
    const status = info.statusBarHeight || 0
    let height = FALLBACK_NAV_HEIGHT
    try {
      const menu = Taro.getMenuButtonBoundingClientRect()
      if (menu && menu.height) {
        height = (menu.top - status) * 2 + menu.height
      }
    } catch {
      // 取不到胶囊（非微信端/调试环境）时退回默认导航高
    }
    return { statusBarHeight: status, navBarHeight: height }
  }, [])

  return (
    <View className='navbar' style={{ paddingTop: `${statusBarHeight}px` }}>
      <View className='navbar-inner' style={{ height: `${navBarHeight}px` }}>
        <Text className='navbar-title'>{title}</Text>
        <View className='navbar-menu' onClick={onMenuClick}>
          <Text className='navbar-menu-icon'>☰</Text>
        </View>
      </View>
    </View>
  )
}
