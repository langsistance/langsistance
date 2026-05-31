'use client'

import type { ReactNode } from 'react'
import { useI18n } from '@/lib/app-i18n'
import { getKnowledgeTypeBadge } from '@/lib/knowledgeTypeBadge'
import {
  getWorkflowInstructionsForReadOnly,
  getWorkflowStepLabels,
  isWorkflowKnowledgeItem,
} from '@/lib/knowledgeWorkflowView'

export interface KnowledgeDetailItem {
  id?: number
  question: string
  answer?: string
  description?: string
  type?: number | string
  params?: string
}

function CloseIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  )
}

export default function KnowledgeDetailModal({
  item,
  onClose,
  footer,
  metadata,
  knowledgeNamesById = {},
}: {
  item: KnowledgeDetailItem
  onClose: () => void
  footer?: ReactNode
  metadata?: ReactNode
  knowledgeNamesById?: Record<number, string>
}) {
  const { t, lang } = useI18n()
  const isWorkflow = isWorkflowKnowledgeItem(item)
  const typeBadge = getKnowledgeTypeBadge(item.type, lang, item.params)
  const stepLabels = getWorkflowStepLabels(item.params, knowledgeNamesById, lang) as string[]
  const instructions = getWorkflowInstructionsForReadOnly(item.answer, stepLabels.length) as string

  return (
    <div className="modal">
      <div className="modal-overlay" onClick={onClose} />
      <div className={`modal-content${isWorkflow ? ' knowledge-editor-modal' : ''}`}>
        <div className="modal-header">
          <h2>{t('modals.knowledgeDetails.title')}</h2>
          <button className="modal-close-btn" onClick={onClose} aria-label={t('modals.close')}>
            <CloseIcon />
          </button>
        </div>
        <div className="modal-body">
          {isWorkflow ? (
            <>
              <div className="form-group">
                <label>{lang === 'en' ? 'Knowledge Type' : '知识类型'}</label>
                <div className="knowledge-readonly-type">
                  <span className={typeBadge.className}>{typeBadge.label}</span>
                </div>
              </div>
              <h4>{t('modals.knowledgeDetails.knowledgeContent')}</h4>
              <div className="form-section">
                <div className="form-group">
                  <label>{lang === 'en' ? 'Name' : '名称'}</label>
                  <input
                    className="form-input knowledge-readonly-input"
                    value={item.question}
                    readOnly
                  />
                </div>
                <div className="form-group">
                  <label>{lang === 'en' ? 'Workflow instructions' : '执行说明'}</label>
                  <textarea
                    className="form-textarea knowledge-readonly-textarea"
                    placeholder={lang === 'en' ? 'No workflow instructions' : '无执行说明'}
                    value={instructions}
                    readOnly
                    rows={4}
                  />
                </div>
                <div className="form-group">
                  <label>{lang === 'en' ? 'Steps' : '步骤知识'}</label>
                  <div className="workflow-step-list">
                    {stepLabels.length > 0 ? (
                      stepLabels.map((label, index) => (
                        <div className="workflow-step-row readonly" key={`${label}-${index}`}>
                          <span className="workflow-step-index">{index + 1}</span>
                          <div className="form-input knowledge-readonly-input workflow-step-readonly">
                            {label}
                          </div>
                        </div>
                      ))
                    ) : (
                      <p className="workflow-step-empty">
                        {lang === 'en' ? 'No workflow steps' : '无步骤知识'}
                      </p>
                    )}
                  </div>
                </div>
                <div className="form-group">
                  <label>{lang === 'en' ? 'Description' : '描述'}</label>
                  <textarea
                    className="form-textarea knowledge-readonly-textarea"
                    placeholder={lang === 'en' ? 'No description' : '无描述'}
                    value={item.description || ''}
                    readOnly
                    rows={2}
                  />
                </div>
              </div>
            </>
          ) : (
            <div className="metadata-section">
              <h4>{t('modals.knowledgeDetails.knowledgeContent')}</h4>
              <div className="detail-item">
                <strong>{t('knowledge.question')}：</strong>{item.question}
              </div>
              {item.answer && (
                <div className="detail-item">
                  <strong>{t('knowledge.answer')}：</strong>{item.answer}
                </div>
              )}
              {item.description && (
                <div className="detail-item">
                  <strong>{lang === 'en' ? 'Description' : '描述'}：</strong>{item.description}
                </div>
              )}
            </div>
          )}
          {metadata && (
            <div className="metadata-section">
              <h4>{t('modals.knowledgeDetails.basicInfo')}</h4>
              {metadata}
            </div>
          )}
        </div>
        <div className="modal-footer">
          {footer}
          <button className="btn btn-secondary" onClick={onClose}>{t('modals.close')}</button>
        </div>
      </div>
    </div>
  )
}
