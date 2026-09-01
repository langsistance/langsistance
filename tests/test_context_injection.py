# -*- coding: utf-8 -*-
"""需求 2 (会话/上下文持久化与注入一致性) 单测。

覆盖:
- _conversation_history_turns / _build_previous_conversation_block:
  请求历史注入、防干扰指令、噪声过滤、当前提问去重、条数上限、截断、
  池内 turns 按 user 隔离兜底
- _read_recent_patent_ids: Redis 最近专利号注入 + 静默降级
- get_or_create_agent: agent 池按 user 键控 (同 user 复用、异 user 隔离、LRU 驱逐)
"""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# api_routes.core 依赖 firebase_admin (未安装) 与 REDIS_* 环境变量
# (sources/user/passport.py 模块级 get_redis_connection())。
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
sys.modules.setdefault("firebase_admin", MagicMock())

from sources.agents.general_agent import (  # noqa: E402
    CONTEXT_TURN_CHARS,
    CONTEXT_TURNS_MAX,
    RELEVANT_TOP_N,
    _build_previous_conversation_block,
    _conversation_history_turns,
    _is_history_noise,
    _read_recent_patent_ids,
)
from api_routes.core import get_or_create_agent  # noqa: E402

# 报告证据回归样本
Q_COMPANY = "帮我分析一下江苏维锂最近的专利"
Q_CAPABILITY = "我能通过codex使用你来批量下载光学镜头专利嘛"
Q_ASSIGNEE = "检索利勃海尔在全世界范围内申请的相关专利"
Q_TIMEFILTER = "我需要巡检机器人底盘专利，时间是2021年到现在的"


class TestHistoryTurns(unittest.TestCase):
    """_conversation_history_turns: 清洗 + 去重 + 上限。"""

    def _hist(self, *msgs):
        return [{"role": r, "content": c} for r, c in msgs]

    def test_filters_noise_and_non_user_roles(self):
        hist = self._hist(
            ("user", "第一问"),
            ("assistant", "🔬 深度研究任务已提交（任务ID: lt_abc），正在后台执行中..."),
            ("system", "system note"),
            ("assistant", "✅ 深度研究任务已完成"),
            ("assistant", "正经回答"),
            ("user", Q_TIMEFILTER),
        )
        turns = _conversation_history_turns(hist, current_query=Q_TIMEFILTER)
        self.assertEqual([t["role"] for t in turns], ["user", "assistant"])
        self.assertEqual(turns[0]["content"], "第一问")
        self.assertEqual(turns[1]["content"], "正经回答")

    def test_drops_current_question_from_history(self):
        # 前端 history 的最后一条 user 消息就是本轮提问 — 不得作为历史注入
        hist = self._hist(("user", "第一问"), ("assistant", "回答"),
                          ("user", Q_ASSIGNEE))
        turns = _conversation_history_turns(hist, current_query=Q_ASSIGNEE)
        self.assertEqual([t["content"] for t in turns], ["第一问", "回答"])

    def test_caps_at_contest_turns_max(self):
        pairs = []
        for i in range(8):
            pairs += [("user", f"q{i}"), ("assistant", f"a{i}")]
        turns = _conversation_history_turns(self._hist(*pairs), "")
        self.assertLessEqual(len(turns), CONTEXT_TURNS_MAX)
        # 保留最近的 N 轮
        self.assertEqual(turns[-1]["content"], "a7")

    def test_truncates_long_content(self):
        long_txt = "长" * (CONTEXT_TURN_CHARS + 50)
        turns = _conversation_history_turns(self._hist(("user", long_txt)), "")
        self.assertLessEqual(len(turns[0]["content"]), CONTEXT_TURN_CHARS + 3)
        self.assertTrue(turns[0]["content"].endswith("..."))

    def test_keeps_patent_ids_from_assistant(self):
        hist = [{"role": "user", "content": "查专利"},
                {"role": "assistant", "content": "结果",
                 "patent_ids": ["8388852", "CN202310123456.7"]}]
        turns = _conversation_history_turns(hist, "")
        self.assertEqual(turns[1]["patent_ids"],
                         ["8388852", "CN202310123456.7"])

    def test_noise_marker(self):
        self.assertTrue(_is_history_noise("🔬 深度研究任务已提交"))
        self.assertTrue(_is_history_noise("Task ID: lt_abc"))
        self.assertTrue(_is_history_noise("❌ 深度研究任务执行失败"))
        self.assertFalse(_is_history_noise("正常回答内容"))


class TestPreviousConversationBlock(unittest.TestCase):
    """_build_previous_conversation_block: 注入 + 防干扰 + 隔离兜底。"""

    def test_request_history_builds_block_with_instruction(self):
        conv_history = [
            {"role": "user", "content": "帮我查一下量子计算专利"},
            {"role": "assistant", "content": "找到以下相关专利",
             "patent_ids": ["12345678"]},
        ]
        block, ids = _build_previous_conversation_block(
            conv_history, [], user_id="u1", current_query=Q_COMPANY)
        self.assertIn("Previous conversation", block)
        self.assertIn("仅作参考", block)           # 防干扰指令
        self.assertIn("LATEST question", block)   # 防干扰指令 (EN)
        self.assertIn("帮我查一下量子计算专利", block)
        self.assertIn("找到以下相关专利", block)
        self.assertIn("前序检索命中专利号（仅当用户引用时使用）：12345678", block)
        self.assertEqual(ids, ["12345678"])

    def test_empty_history_returns_empty(self):
        block, ids = _build_previous_conversation_block([], [], "u1", "q")
        self.assertEqual((block, ids), ("", []))

    def test_fallback_pooled_turns_filtered_by_user(self):
        pooled = [
            {"user": "u1的提问", "assistant": "u1的回答", "user_id": "u1"},
            {"user": "u2的提问", "assistant": "u2的回答", "user_id": "u2"},
        ]
        block, _ = _build_previous_conversation_block(
            None, pooled, user_id="u1", current_query="u1的新问题")
        self.assertIn("u1的提问", block)
        self.assertNotIn("u2的提问", block)   # 跨用户隔离

    def test_legacy_turns_without_user_id_kept_as_current_user(self):
        pooled = [{"user": "旧提问", "assistant": "旧回答"}]  # 无 user_id
        block, _ = _build_previous_conversation_block(
            None, pooled, user_id="u1", current_query="新问题")
        self.assertIn("旧提问", block)


class TestReadRecentPatentIds(unittest.TestCase):
    """_read_recent_patent_ids: Redis 跨会话专利号注入 + 静默降级。"""

    @patch("sources.agents.general_agent.get_redis_connection")
    def test_reads_stored_ids(self, mock_redis):
        mock_redis.return_value.get.return_value = '["8388852", "CN202310123456.7"]'
        self.assertEqual(_read_recent_patent_ids("u1"),
                         ["8388852", "CN202310123456.7"])

    @patch("sources.agents.general_agent.get_redis_connection")
    def test_empty_when_no_stored(self, mock_redis):
        mock_redis.return_value.get.return_value = None
        self.assertEqual(_read_recent_patent_ids("u1"), [])

    @patch("sources.agents.general_agent.get_redis_connection")
    def test_silent_degrade_on_redis_error(self, mock_redis):
        mock_redis.side_effect = RuntimeError("redis down")
        self.assertEqual(_read_recent_patent_ids("u1"), [])

    @patch("sources.agents.general_agent.get_redis_connection")
    def test_silent_degrade_on_bad_json(self, mock_redis):
        mock_redis.return_value.get.return_value = "not-json"
        self.assertEqual(_read_recent_patent_ids("u1"), [])


class TestAgentPoolUserKeyed(unittest.TestCase):
    """get_or_create_agent: 池按 user 键控 — 追问必命中同一实例。"""

    def setUp(self):
        import api_routes.core as core_mod
        self._core = core_mod
        core_mod._agent_pool.clear()
        self._logger = MagicMock()

    async def _factory(self):
        return object()

    def test_same_user_reuses_same_instance(self):
        agent1 = _run(get_or_create_agent("u1", self._factory, self._logger))
        agent2 = _run(get_or_create_agent("u1", self._factory, self._logger))
        self.assertIs(agent1, agent2)

    def test_different_users_get_distinct_instances(self):
        a1 = _run(get_or_create_agent("u1", self._factory, self._logger))
        a2 = _run(get_or_create_agent("u2", self._factory, self._logger))
        self.assertIsNot(a1, a2)
        # 各自稳定复用
        self.assertIs(a1, _run(get_or_create_agent("u1", self._factory, self._logger)))
        self.assertIs(a2, _run(get_or_create_agent("u2", self._factory, self._logger)))

    def test_pool_full_evicts_lru_user(self):
        pool_size = self._core.AGENT_POOL_MAX_SIZE  # 3 (2C2G 内存预算)
        agents = {f"u{i}": _run(get_or_create_agent(f"u{i}", self._factory, self._logger))
                  for i in range(1, pool_size + 1)}  # 池满
        # 把 u1 的实例时间戳调旧 (但未过期), 使其成为 LRU
        self._core._agent_pool["u1"] = (
            agents["u1"], datetime.now() - timedelta(minutes=4))
        # 新用户挤占: u1 被驱逐, 槽位重新键控给 u_new
        new_agent = _run(get_or_create_agent("u_new", self._factory, self._logger))
        self.assertIsNot(new_agent, agents["u1"])
        # u1 再来: 原实例已被驱逐 → 新实例 (此时 LRU 变为 u2, 被挤掉);
        # 且绝不能拿到 u_new 的实例 (驱逐必须重新键控, 否则破坏 user-keyed)
        u1_again = _run(get_or_create_agent("u1", self._factory, self._logger))
        self.assertIsNot(u1_again, agents["u1"])
        self.assertIsNot(u1_again, new_agent)
        # u_new 的实例仍归 u_new
        self.assertIs(new_agent,
                      _run(get_or_create_agent("u_new", self._factory, self._logger)))
        # 最近使用的用户不受影响 (u{pool_size} 是 fill 循环里最新创建的)
        last_key = f"u{pool_size}"
        self.assertIs(agents[last_key],
                      _run(get_or_create_agent(last_key, self._factory, self._logger)))


def _run(coro):
    import asyncio
    return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()


class TestDeliveryFormatGuidance(unittest.TestCase):
    """需求 3: 检索答复必须带"最高关联专利"结论格式约束。"""

    def _agent(self):
        from unittest.mock import MagicMock
        from sources.agents.general_agent import GeneralAgent
        agent = GeneralAgent.__new__(GeneralAgent)
        agent.logger = MagicMock()
        agent.llm = MagicMock()
        return agent

    def test_loop_guidance_requires_top_matching_patent(self):
        agent = self._agent()
        guidance = agent._loop_system_guidance()
        self.assertIn("Top matching result", guidance)
        self.assertIn("MANDATORY", guidance)
        self.assertIn("complete and copyable", guidance)
        self.assertIn(str(RELEVANT_TOP_N), guidance)
        # 通用指令 — 不得固化任何测试提问词
        self.assertNotIn("干燥空气", guidance)
        self.assertNotIn("RGB", guidance)


class TestNotLoggedInPrompt(unittest.TestCase):
    """需求 4-2: 未认证提示可操作化 — 不得再指示 LLM 输出内部标记。"""

    def _agent(self):
        from unittest.mock import MagicMock
        from sources.agents.general_agent import GeneralAgent
        agent = GeneralAgent.__new__(GeneralAgent)
        agent.logger = MagicMock()
        agent.llm = MagicMock()
        agent.knowledgeTool = (None, None)  # not tool_info 分支
        return agent

    def test_no_internal_marker_in_prompts(self):
        agent = self._agent()
        for prompt in (
            agent.generate_fixed_system_prompt(),
            agent.generate_template_system_prompt(),
        ):
            self.assertNotIn("<Knowledge tool not logged in>", prompt)
            self.assertIn("需要登录", prompt)
            self.assertIn("Do NOT output internal markers", prompt)
