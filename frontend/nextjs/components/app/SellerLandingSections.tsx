'use client'

/**
 * 卖家安全台模式的空态内容（输入框下方）：
 * 「卖家的每一步，都有人替你盯着专利」五环节 + 「六大模块」。
 * 文案与原型 v3 一致（用户已认可），仅通用产品文案，不含单条测试提问。
 */

const STAGES = [
  { no: '1', title: '选品排雷', desc: '看中一个产品，先查有没有人在先布局' },
  { no: '2', title: '上架自查', desc: '风险报告：改哪一处能规避，写清楚' },
  { no: '3', title: '日常盯梢', desc: '盯竞品新专利，防爆款被抢注' },
  { no: '4', title: '供应商验真', desc: '"这是我们的专利产品"？一分钟验证' },
  { no: '5', title: '被诉应对', desc: '投诉后 48 小时，先看懂对方专利' },
]

const MODULES = [
  { mark: '查', tone: 'bg-teal-50 text-teal-700', tag: '已上线', tagTone: 'bg-teal-50 text-teal-700', name: '查一查', desc: '产品图片 / 产品名 / ASIN，中美专利库找疑似相关，按相似度排序。' },
  { mark: '卡', tone: 'bg-teal-50 text-teal-700', tag: '已上线', tagTone: 'bg-teal-50 text-teal-700', name: '专利卡', desc: '任意专利号一屏说清：保护什么、撞不撞、到期没、下一步。' },
  { mark: '报', tone: 'bg-amber-50 text-amber-700', tag: 'M2', tagTone: 'bg-amber-50 text-amber-700', name: '风险报告', desc: '逐件比对 + 风险等级 + 行动建议，PDF 带署名水印可分享。' },
  { mark: '真', tone: 'bg-amber-50 text-amber-700', tag: 'M2', tagTone: 'bg-amber-50 text-amber-700', name: '供应商验真', desc: '供应商的专利号：是真的吗、还有效吗、真覆盖这个产品吗？' },
  { mark: '盯', tone: 'bg-gray-100 text-gray-500', tag: 'M3', tagTone: 'bg-gray-100 text-gray-400', name: '盯一盯', desc: '竞品新专利、相似外观公开、专利到期，邮件推送。防抢注靠它。' },
  { mark: '救', tone: 'bg-red-50 text-red-600', tag: 'M3', tagTone: 'bg-gray-100 text-gray-400', name: '投诉应对包', desc: '对方专利人话解读 + 弱点提示 + 申诉要点草稿（中英双语）。' },
]

export default function SellerLandingSections() {
  return (
    <div className="w-full max-w-3xl px-4 mt-10 text-left">
      <h3 className="text-lg font-bold text-gray-900 text-center">
        卖家的每一步，都有人替你盯着专利
      </h3>
      <p className="text-center text-xs text-gray-500 mt-1 mb-6">
        覆盖选品到被诉的完整生命周期
      </p>
      <div className="grid grid-cols-5 gap-2">
        {STAGES.map((stage) => (
          <div key={stage.no} className="text-center px-1">
            <span className="inline-flex items-center justify-center w-7 h-7 rounded-full border border-gray-300 text-teal-700 text-xs font-semibold">
              {stage.no}
            </span>
            <p className="mt-2 text-[13px] font-semibold text-gray-900">{stage.title}</p>
            <p className="mt-0.5 text-[11px] leading-relaxed text-gray-500">{stage.desc}</p>
          </div>
        ))}
      </div>

      <h3 className="text-lg font-bold text-gray-900 text-center mt-12">
        六大模块，免费起步
      </h3>
      <p className="text-center text-xs text-gray-500 mt-1 mb-6">
        所有分析标注检索范围与免责声明——不藏不确定性
      </p>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {MODULES.map((mod) => (
          <div
            key={mod.name}
            className="rounded-xl border border-gray-200 p-4 flex gap-3 items-start"
          >
            <span
              className={`w-8 h-8 rounded-lg grid place-items-center font-bold text-sm flex-shrink-0 ${mod.tone}`}
            >
              {mod.mark}
            </span>
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <h4 className="text-sm font-semibold text-gray-900">{mod.name}</h4>
                <span className={`text-[10px] rounded-full px-2 py-0.5 font-semibold ${mod.tagTone}`}>
                  {mod.tag}
                </span>
              </div>
              <p className="text-xs text-gray-500 mt-1 leading-relaxed">{mod.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
