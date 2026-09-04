'use client'

/**
 * 卖家安全台空态区块（输入框下方）：五环节 + 六大模块。
 *
 * 视觉与既有 chat-landing 卡片体系一致（同一套圆角卡 / 色板 / 图标格），
 * 文案只留一句话级，与专业模式六能力卡同密度。
 */

const STAGES = [
  { no: '1', title: '选品排雷' },
  { no: '2', title: '上架自查' },
  { no: '3', title: '日常盯梢' },
  { no: '4', title: '供应商验真' },
  { no: '5', title: '被诉应对' },
]

// tag: '' = 已上线（teal），'soon' = M2/M3（amber 弱化）
const MODULES = [
  { mark: '查', tag: '', name: '查一查', desc: '图片 / 名称 / ASIN 找疑似专利' },
  { mark: '卡', tag: '', name: '专利卡', desc: '专利号一屏读懂四件事' },
  { mark: '报', tag: 'soon', name: '风险报告', desc: '逐件比对，PDF 可分享' },
  { mark: '真', tag: 'soon', name: '供应商验真', desc: '证书真伪一分钟判断' },
  { mark: '盯', tag: 'soon', name: '盯一盯', desc: '专利公开与到期提醒' },
  { mark: '救', tag: 'soon', name: '投诉应对包', desc: '对方专利解读与申诉要点' },
]

const TAG_TEXT: Record<string, string> = { '': '已上线', soon: '规划中' }

export default function SellerLandingSections() {
  return (
    <div className="chat-landing-section flex flex-col items-center gap-6 pt-4">
      <div className="flex flex-col items-center gap-2">
        <h3 className="seller-landing-h">卖家的每一步，都有人替你盯着专利</h3>
        <p className="seller-landing-sub">选品到被诉，全生命周期</p>
      </div>
      <div className="seller-stages">
        {STAGES.map((stage) => (
          <span key={stage.no} className="seller-stage">
            <b>{stage.no}</b>
            {stage.title}
          </span>
        ))}
      </div>

      <h3 className="seller-landing-h pt-2">六大模块，免费起步</h3>
      <div className="seller-landing-modules">
        {MODULES.map((mod) => (
          <div
            key={mod.name}
            className={`chat-landing-card${mod.tag === 'soon' ? ' free' : ''}`}
          >
            <div className="chat-landing-card-icon" aria-hidden="true">
              <span style={{ fontSize: 16, fontWeight: 700 }}>{mod.mark}</span>
            </div>
            <div className="chat-landing-card-body" style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <h4 className="chat-landing-card-title" style={{ marginBottom: 0 }}>
                  {mod.name}
                </h4>
                <span className={`seller-tag${mod.tag === 'soon' ? ' soon' : ''}`}>
                  {TAG_TEXT[mod.tag]}
                </span>
              </div>
              <p className="chat-landing-card-desc">{mod.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
