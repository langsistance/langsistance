-- ============================================================================
-- UPDATE batch patent analysis knowledge — zh:|en: bilingual format
-- with natural user-question style descriptions
-- ============================================================================

UPDATE knowledge
SET
    description = 'zh:基于专利大模型的批量专利分析：
• 搜索分析 — 帮我看看特斯拉近期在自动驾驶领域有什么新专利
• 指定专利分析 — 分析专利 17429113、18012525、18331482
• 文件上传分析 — 从这些文件中筛选出人工智能相关专利
• 追问分析结果 — 这里面哪些是和人工智能有关的专利|en:Batch patent analysis powered by patent LLM:
• Search Analysis — What recent patents does Tesla have in autonomous driving
• Designated Patents — Analyze patents 17429113, 18012525, 18331482
• File Upload — Filter these documents for AI-related patents
• Follow-up — Which of these are AI-related patents'
WHERE params = '{"type": "long_task"}'
  AND description LIKE '%批量专利%'
LIMIT 1;