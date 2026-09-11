import { Text, View } from '@tarojs/components'
import { TaskState, isTerminal, phaseLabel } from '../../services/longTask'
import './index.scss'

type Props = {
  task: TaskState
  onRetry: () => void
  onDownload: (format: string) => void
}

export default function LongTaskCard({ task, onRetry, onDownload }: Props) {
  const done = task.status === 'completed'
  const failed = task.status === 'failed' || task.status === 'cancelled'
  const pct = Math.max(0, Math.min(100, task.progress || 0))

  return (
    <View className='ltcard'>
      <View className='ltcard-head'>
        <Text className='ltcard-title'>
          {done ? '分析完成' : failed ? '分析失败' : '正在分析'}
        </Text>
        {!isTerminal(task.status) ? (
          <Text className='ltcard-pct'>{pct}%</Text>
        ) : null}
      </View>

      {!isTerminal(task.status) ? (
        <>
          <View className='ltcard-bar'>
            <View className='ltcard-bar-fill' style={{ width: `${pct}%` }} />
          </View>
          <Text className='ltcard-step'>
            {task.step || phaseLabel(task.phase)}
          </Text>
        </>
      ) : null}

      {failed ? (
        <>
          <Text className='ltcard-error'>{task.error || '任务未能完成'}</Text>
          <View className='ltcard-retry' onClick={onRetry}>
            <Text>重试</Text>
          </View>
        </>
      ) : null}

      {done && task.reportFiles.length > 0 ? (
        <View className='ltcard-files'>
          {task.reportFiles.map((f) => (
            <View
              key={f.format}
              className='ltcard-file'
              onClick={() => onDownload(f.format)}
            >
              <Text className='ltcard-file-badge'>
                {f.format.toUpperCase()}
              </Text>
              <Text className='ltcard-file-label'>下载报告</Text>
            </View>
          ))}
        </View>
      ) : null}
    </View>
  )
}
