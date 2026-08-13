'use client'

import { useI18n } from '@/lib/app-i18n'
import DocTab from './DocTab'
import SpecTab from './SpecTab'
import ClaimsTab from './ClaimsTab'
import ProsecutionTab from './ProsecutionTab'

const TAB_ORDER = ['details', 'doc', 'spec', 'claims', 'prosecution']

export default function DetailPanel({
  row, tab, onTabChange,
}: {
  row: any
  tab: string
  onTabChange: (tab: string) => void
}) {
  const { t } = useI18n()
  if (!row) {
    return <div className="results-detail-empty">← {t('results.emptyHint')}</div>
  }
  const availableTabs = TAB_ORDER.filter((key) => {
    if (key === 'doc') return row.isDocument
    if (key === 'details') return !row.isDocument
    if (key === 'prosecution') return !row.isDocument
    return !row.isDocument
  })
  const activeTab = availableTabs.includes(tab) ? tab : availableTabs[0]

  return (
    <div className="results-detail">
      <div className="results-detail-tabs">
        {availableTabs.map((key) => (
          <button
            key={key}
            className={activeTab === key ? 'active' : ''}
            onClick={() => onTabChange(key)}
          >
            {t(`results.tab${key.charAt(0).toUpperCase() + key.slice(1)}`)}
          </button>
        ))}
      </div>
      <div className="results-detail-body">
        {activeTab === 'details' && (
          <div className="results-detail-card">
            <h3>{row.title}</h3>
            {row.fields.length > 0 ? (
              <table className="results-field-table">
                <tbody>
                  {row.fields.map(([label, value]: [string, string]) => (
                    <tr key={label}>
                      <td className="results-field-label">{label}</td>
                      <td>{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p>{t('results.fieldTableEmpty')}</p>
            )}
          </div>
        )}
        {activeTab === 'doc' && <DocTab row={row} />}
        {activeTab === 'spec' && <SpecTab row={row} />}
        {activeTab === 'claims' && <ClaimsTab row={row} />}
        {activeTab === 'prosecution' && <ProsecutionTab row={row} />}
      </div>
    </div>
  )
}
