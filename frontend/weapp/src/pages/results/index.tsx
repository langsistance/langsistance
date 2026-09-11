import { useMemo, useState } from 'react'
import { ScrollView, Text, View } from '@tarojs/components'
import { useRouter } from '@tarojs/taro'
import { resultsStore } from '../../services/resultsStore'
import { ResultsPayload, metaLine, pickColumn } from '../../utils/results'
import ResultDetail from '../../components/ResultDetail'
import './index.scss'

/**
 * 专利结果列表。数据来自 resultsStore：
 * 本会话的结果在内存里（全量），重开小程序后从 storage 读回（裁剪版 ≤40 行）。
 */
export default function ResultsPage() {
  const router = useRouter()
  const setId = decodeURIComponent(String(router.params.set || ''))

  // 先查内存（当前会话全量），再回落到 storage（历史裁剪版）
  const payload = useMemo<ResultsPayload | null>(
    () => resultsStore.get(setId) || resultsStore.load(setId),
    [setId],
  )

  const [activeIndex, setActiveIndex] = useState(-1)

  if (!payload || payload.rows.length === 0) {
    return (
      <View className='results-empty'>
        <Text className='results-empty-title'>结果已不可用</Text>
        <Text className='results-empty-hint'>
          本机没有这份结果的缓存（可能换了设备或清理过缓存）。请重新发起检索。
        </Text>
      </View>
    )
  }

  const row = activeIndex >= 0 ? payload.rows[activeIndex] : null

  return (
    <View className='results'>
      <ScrollView className='results-scroll' scrollY>
        <View className='results-count'>共 {payload.rows.length} 项</View>
        {payload.rows.map((r, i) => {
          const title = pickColumn(payload, 'title', r) || '—'
          const meta = metaLine(payload, r)
          return (
            <View
              key={i}
              className='results-row'
              onClick={() => setActiveIndex(i)}
            >
              <View className='results-row-main'>
                <Text className='results-row-title'>{title}</Text>
                {meta.length > 0 ? (
                  <Text className='results-row-meta'>{meta.join(' · ')}</Text>
                ) : null}
              </View>
              <Text className='results-row-arrow'>›</Text>
            </View>
          )
        })}
      </ScrollView>

      {row ? (
        <ResultDetail
          payload={payload}
          row={row}
          visible
          onClose={() => setActiveIndex(-1)}
        />
      ) : null}
    </View>
  )
}
