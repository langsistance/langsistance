-- ============================================================================
-- UPDATE existing USPTO prosecution knowledge to use zh:|en: bilingual format
-- Find the existing record by params = '{"type": "long_task"}' + question pattern
-- ============================================================================

UPDATE knowledge
SET
    question = 'zh:输入美国专利申请号，分析其审查历史（USPTO）|en:Enter a US patent application number to analyze its prosecution history',
    description = 'zh:USPTO审查历史分析（美国专利）：
• 无效分析 — 帮我看看专利US12506212的主要发明点和File History
• 驳回答复 — 我刚收到US12506212的非最终驳回，帮我分析驳回理由和答复策略
• 修改技巧 — 专利US12506212审查中修改了几次，分析修改的技巧和目的
• 侵权抗辩 — 审查档案中是否有可用于侵权抗辩的权利要求限制陈述
• 最新动态 — 专利US12506212最近有没有新的审查意见或授权公告
• 术语解释 — 权利要求中的这个术语在审查历史中是如何定义或限制的|en:USPTO prosecution history analysis:
• Invalidation Analysis — Review key features and file history of patent US12506212
• Office Action Response — I received a non-final rejection for US12506212, analyze and propose response
• Amendment Review — Review amendment techniques used during prosecution of US12506212
• Infringement Defense — Find limiting statements in prosecution record for infringement defense
• Status Tracking — Any recent office actions or notice of allowance for US12506212
• Term Interpretation — How was this claim term defined or limited during examination',
    answer = 'zh:自动从USPTO下载审查文件（Office Action、Applicant Response、Amendment、Notice of Allowance等）并使用AI生成详细分析报告。仅适用于USPTO美国专利（8位数字申请号）。|en:Automatically downloads prosecution documents from USPTO (Office Actions, Applicant Responses, Amendments, Notices of Allowance) and generates detailed AI analysis reports. US patents only (8-digit application numbers).'
WHERE params = '{"type": "long_task"}'
  AND question LIKE '%USPTO%'
  AND question NOT LIKE '%zh:%'
LIMIT 1;