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
    '根据专利申请号分析审查历史（美国USPTO 8位申请号）',
    '示例：分析专利 17429113 的审查历史。'
    '适用场景：用户提供美国专利申请号（8位纯数字，如17429113），要求分析审查过程、Office Action、答辩策略、Claim修改、授权原因等。'
    '系统自动从USPTO下载审查文件（Office Action、Applicant Response、Amendment、Notice of Allowance），'
    '使用AI生成详细的审查历史分析报告。'
    '注意：仅适用于USPTO美国专利（8位数字申请号），中国专利审查（CN开头的申请号）请使用其他工具。',
    '使用USPTO审查历史分析功能，自动下载审查文件并生成AI分析报告。此知识为long_task类型，选中后将触发后台专利审查分析任务。',
    2,   -- public=2: visible to all users
    1,   -- status=1: active
    0,   -- embedding_id: auto-generated on first search
    '',  -- model_name: auto-set by system
    0,   -- tool_id: no specific tool, long_task pipeline handles everything
    '{"type": "long_task"}'   -- params with type=long_task → type=3 knowledge
);
