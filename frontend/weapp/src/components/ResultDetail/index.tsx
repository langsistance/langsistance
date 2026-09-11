import { useState } from 'react'
import { ScrollView, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import { ResultsPayload, pickColumn } from '../../utils/results'
import { ClaimItem, fetchClaims, fetchSpec } from '../../services/patentDetail'
import { openOrShareFile } from '../../services/download'
import { errorText } from '../../services/api'
import './index.scss'

type Tab = 'details' | 'spec' | 'claims'

/**
 * `status` 一列不进详情字段表。上游 react_tools.py 的候选行带着它（值多为空串），
 * 裁剪后仍留在 columns/rows 里；`status` 的取值语义**未核实**（web 的
 * ClaimsTab 拿它当 active/canceled 用，与候选行的含义未必同源），故不猜、不渲染。
 */
const HIDDEN_FIELD_KEYS = ['status']

type Props = {
  payload: ResultsPayload
  row: Record<string, string>
  visible: boolean
  onClose: () => void
}

export default function ResultDetail({ payload, row, visible, onClose }: Props) {
  const [tab, setTab] = useState<Tab>('details')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [claims, setClaims] = useState<ClaimItem[] | null>(null)
  const [claimsPdf, setClaimsPdf] = useState('')

  if (!visible) return null

  const patentId = pickColumn(payload, 'patent_id', row)
  const appNumber = pickColumn(payload, 'application_number', row)
  // 行级 source 优先于 payload 级（对齐 web results.js:66-70）
  const rowSource = String(row.source || '') || payload.source
  const title = pickColumn(payload, 'title', row) || '—'
  const canQuery = Boolean(patentId || appNumber)

  // 详情的字段表：全部**非空**字段，标签用列的中文 label
  const fields = payload.columns
    .filter((c) => HIDDEN_FIELD_KEYS.indexOf(c.key) < 0)
    .map((c) => ({ label: c.label, value: String(row[c.key] ?? '') }))
    .filter((f) => f.value)

  /**
   * 下载 PDF 到临时文件后打开。
   * 不自己调 openDocument——复用 M3 的 openOrShareFile，它已经带了
   * 「打不开就回退到 shareFileMessage 转发」的分支（部分机型对 office/PDF
   * 支持不全）。
   */
  async function openPdf(url: string) {
    const res = await Taro.downloadFile({ url })
    if (res.statusCode >= 400) throw new Error('文件下载失败')
    await openOrShareFile(res.tempFilePath, 'pdf')
  }

  async function loadSpec() {
    if (busy) return
    setBusy(true)
    setError('')
    try {
      const r = await fetchSpec(rowSource, patentId, appNumber)
      if (!r.ok) {
        setError(r.message)
        return
      }
      await openPdf(r.pdfUrl)
    } catch (err) {
      setError(errorText(err, '说明书加载失败'))
    } finally {
      setBusy(false)
    }
  }

  async function loadClaims() {
    if (busy) return
    setBusy(true)
    setError('')
    try {
      const r = await fetchClaims(rowSource, patentId, appNumber)
      if (!r.ok) {
        setError(r.message)
        return
      }
      setClaims(r.claims)
      setClaimsPdf(r.pdfUrl)
      // 后端没给结构化权利要求 → 直接开 PDF
      if (r.claims.length === 0 && r.pdfUrl) {
        await openPdf(r.pdfUrl)
      }
    } catch (err) {
      setError(errorText(err, '权利要求加载失败'))
    } finally {
      setBusy(false)
    }
  }

  function switchTab(next: Tab) {
    setTab(next)
    setError('')
    if (next === 'spec' && claims === null) loadSpec()
    if (next === 'claims' && claims === null) loadClaims()
  }

  return (
    <View className='rd'>
      <View className='rd-head'>
        <View className='rd-back' onClick={onClose}>
          <Text className='rd-back-icon'>‹</Text>
        </View>
        <View className='rd-tabs'>
          {(['details', 'spec', 'claims'] as Tab[]).map((key) => (
            <View
              key={key}
              className={`rd-tab${tab === key ? ' rd-tab-active' : ''}`}
              onClick={() => switchTab(key)}
            >
              <Text>
                {key === 'details' ? '详情' : key === 'spec' ? '说明书' : '权利要求'}
              </Text>
            </View>
          ))}
        </View>
      </View>

      {!canQuery && tab !== 'details' ? (
        <View className='rd-note'>
          <Text>该条缺少可查询的专利号</Text>
        </View>
      ) : null}

      {busy ? (
        <View className='rd-note'>
          <Text>加载中…</Text>
        </View>
      ) : null}

      {error ? (
        <View className='rd-error'>
          <Text>{error}</Text>
        </View>
      ) : null}

      <ScrollView className='rd-body' scrollY>
        {tab === 'details' ? (
          <View className='rd-fields'>
            <Text className='rd-title'>{title}</Text>
            {fields.map((f) => (
              <View key={f.label} className='rd-field'>
                <Text className='rd-field-label'>{f.label}</Text>
                <Text className='rd-field-value'>{f.value}</Text>
              </View>
            ))}
          </View>
        ) : null}

        {tab === 'claims' && claims && claims.length > 0 ? (
          <View className='rd-claims'>
            {claims.map((c) => (
              <View key={c.number} className='rd-claim'>
                <Text className='rd-claim-head'>
                  第 {c.number} 项 · {c.independent ? '独立' : '从属'}
                </Text>
                <Text className='rd-claim-text'>{c.text}</Text>
              </View>
            ))}
          </View>
        ) : null}

        {tab === 'claims' && claims && claims.length === 0 && claimsPdf ? (
          <View className='rd-note'>
            <Text>该专利无结构化权利要求，已打开 PDF</Text>
          </View>
        ) : null}
      </ScrollView>
    </View>
  )
}
