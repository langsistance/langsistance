-- ============================================================================
-- Add USPTO Prosecution History Analysis knowledge entry (type=3, long_task)
-- ============================================================================
-- This knowledge triggers the execute_prosecution_analysis Celery task when
-- a user asks to analyze a USPTO patent's prosecution/examination history.
--
-- How it works:
--   1. User asks e.g. "帮我分析一下专利 17429113 的审查历史"
--   2. Knowledge base search finds this entry (type=3 always a candidate)
--   3. Routing LLM picks this entry (matches "审查历史" + USPTO context)
--   4. Type=3 triggers long_task intent → _classify_long_task_async()
--   5. LLM classifies as "prosecution" scenario → execute_prosecution_analysis
-- ============================================================================

INSERT INTO knowledge (
    user_id,
    question,
    description,
    answer,
    public,
    status,
    embedding_id,
    model_name,
    tool_id,
    params
) VALUES (
    0,   -- system-level knowledge (public=2 means visible to all users)
    'zh:输入美国专利申请号，分析其审查历史（USPTO）|en:Enter a US patent application number to analyze its prosecution history',
    'zh:USPTO审查历史分析（美国专利）：
• 无效分析 — 帮我看看专利US12506212的主要发明点和File History
• 驳回答复 — 我刚收到US12506212的非最终驳回，帮我分析驳回理由和答复策略
• 修改技巧 — 专利US12506212审查中修改了几次，分析修改的技巧和目的
• 侵权抗辩 — 审查档案中是否有可用于侵权抗辩的权利要求限制陈述
• 最新动态 — 专利US12506212最近有没有新的审查意见或授权公告|en:USPTO prosecution history analysis:
• Invalidation Analysis — Review key features and file history of patent US12506212
• Office Action Response — I received a non-final rejection for US12506212, analyze and propose response
• Amendment Review — Review amendment techniques used during prosecution of US12506212
• Infringement Defense — Find limiting statements in prosecution record for infringement defense
• Status Tracking — Any recent office actions or notice of allowance for US12506212',
    'zh:自动从USPTO下载审查文件（Office Action、Applicant Response、Amendment、Notice of Allowance等）并使用AI生成详细分析报告。仅适用于USPTO美国专利（8位数字申请号）。|en:Automatically downloads prosecution documents from USPTO (Office Actions, Applicant Responses, Amendments, Notices of Allowance) and generates detailed AI analysis reports. US patents only (8-digit application numbers).',
    2,   -- public=2: visible to all users
    1,   -- status=1: active
    0,   -- embedding_id: auto-generated on first search
    '',  -- model_name: auto-set by system
    0,   -- tool_id: no specific tool, long_task pipeline handles everything
    '{"type": "long_task"}'   -- params with type=long_task → type=3 knowledge
);