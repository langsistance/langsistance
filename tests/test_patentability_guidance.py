# -*- coding: utf-8 -*-
"""需求#28: patentability / allowability consultation → structured delivery.

A query like "如果申请一个以X为核心的发明，是否有授权可能性" must get a
structured assessment template appended to its system prompt (closest
prior art → distinguishing features → inventive-step risk → filing
direction → disclaimer), while ordinary searches / legal-status /
licensing questions keep the plain Search-Result-Delivery format.
"""
import sys
import unittest

# general_agent 依赖 passport/知识库在导入期初始化 — 与 test_long_task_fallback
# 同款桩法（仅桩 passport，其他保持真实）。
import types as _types
_fake_passport = _types.ModuleType("sources.user.passport")
_fake_passport.verify_firebase_token = lambda *a, **k: {"uid": "1"}
_fake_passport.check_and_increase_usage = lambda *a, **k: True
_fake_passport.ensure_local_user_record = lambda *a, **k: None
sys.modules.setdefault("sources.user.passport", _fake_passport)

from sources.agents.general_agent import (  # noqa: E402
    _is_patentability_request,
    _patentability_template_guidance,
)


class TestPatentabilityPredicate(unittest.TestCase):
    def test_assessment_phrasing_hits(self):
        for q in (
            "如果申请一个可折叠桌子的发明，是否有授权可能性？",
            "以标准化箱体为存衣核心的设计能否获得授权",
            "这个技术方案可专利性怎么样",
            "我的产品设计申请专利会不会被驳回",
            "这样一个结构能不能申请专利",
            "申请前景如何？该装置申请发明专利有望吗？",
        ):
            self.assertTrue(_is_patentability_request(q), q)

    def test_legal_status_retrieval_does_not_hit(self):
        for q in (
            "帮我查一下 CN114948588A 的授权状态",
            "这个专利是否已授权了？",
            "查这家公司近一年授权的专利",
            "检索利勃海尔在全世界范围内申请的相关专利",
            "这个专利的授权时间是什么时候",
        ):
            self.assertFalse(_is_patentability_request(q), q)

    def test_licensing_and_enforcement_do_not_hit(self):
        for q in (
            "这个专利能否授权给他人实施",
            "把专利许可给第三方要什么手续",
            "这款产品会不会侵犯别人的专利",
        ):
            self.assertFalse(_is_patentability_request(q), q)

    def test_empty_and_plain_search_do_not_hit(self):
        self.assertFalse(_is_patentability_request(""))
        self.assertFalse(_is_patentability_request("找一下相关专利"))
        self.assertFalse(_is_patentability_request(None))


class TestPatentabilityTemplate(unittest.TestCase):
    def test_zh_has_major_sections(self):
        text = _patentability_template_guidance("zh")
        for anchor in ("区别技术特征", "创造性/显而易见性风险",
                       "申请方向", "免责声明", "真实检索", "禁止编造"):
            self.assertIn(anchor, text)

    def test_en_has_major_sections(self):
        text = _patentability_template_guidance("en")
        for anchor in ("closest prior", "distinguishing",
                       "obviousness", "Disclaimer:", "Never cite"):
            self.assertIn(anchor, text)

    def test_no_business_vocabulary(self):
        # 铁律 [[reject-query-specific-synonym-hardcoding]] — the template
        # must not embed any concrete user/domain query wording.
        zh = _patentability_template_guidance("zh")
        for banned in ("衣柜", "箱体", "桌子", "智能"):
            self.assertNotIn(banned, zh)


class TestGuidanceFor(unittest.TestCase):
    """The seam helper create_agent calls — template only when predicate hits."""

    def test_patentability_prompt_gains_template(self):
        from sources.agents.general_agent import _patentability_guidance_for
        out = _patentability_guidance_for(
            "如果申请一个可折叠椅子的发明，是否有授权可能性？", "zh")
        self.assertIn("可专利性/授权前景评估交付格式", out)

    def test_plain_search_prompt_without_template(self):
        from sources.agents.general_agent import _patentability_guidance_for
        out = _patentability_guidance_for("找一下可折叠椅子相关的专利", "zh")
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
