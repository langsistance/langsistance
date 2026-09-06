"""test_design_pipeline: design_pipeline.run 编排 (design P1 T7)。

6 场景全 mock (禁真实网络)。说明: L1/L2 的网络 fetch 由模块级 mock 取代;
流程①-⑥ 逐一驱动 run 以验证 digest/report/进度与异常分流。
"""
import asyncio
from unittest import mock

import pytest

from sources.design.design_pipeline import (DesignNeedClarification,
                                            DesignRuntimeError, run)
from sources.design.design_risk import DesignCandidate, JudgeVerdict

PARAMS = {"product_image_refs": ["p.jpg"], "product_text": "",
          "source": "us_design"}


class _Ctx:
    def __init__(self):
        self.progress_msgs = []

    def progress(self, msg):
        self.progress_msgs.append(msg)

    def warning(self, msg):
        self.progress_msgs.append(f"WARN:{msg}")


def _cand(pub="USD1A", title="Toy", date="2023-01-01", state="active"):
    return DesignCandidate(pub, title, date, "ACME", state)


def _search(cands, limited=False):
    async def f(fetch, en_name, keywords, country_param="", jitter=1.0):
        return {"candidates": list(cands), "em_cn": [], "used_queries": 1,
                "rate_limited": limited}
    return f


async def _ok_pdf(fetch_page, pid):
    return b"%PDF-1.4 fake small"


async def _raise_pdf(fetch_page, pid):
    raise RuntimeError("L2-connect")


async def _judge_ok(product_images, pdf_by, call=None):
    return [JudgeVerdict("USD1A", 0.82, "high",
                         (("整体轮廓", 0.8, "一致"),),
                         "蛇头近似", "尾部开关不同")]


async def _judge_empty(product_images, pdf_by, call=None):
    return []


def _patch(design_query, search, pdf=None, judge=None):
    """design_query: 固定 L0 dict; search: async f; pdf: async f(2 参数); judge: async."""
    if design_query is None:
        design_query = {"en_name": "Toy", "keywords": ["toy"],
                        "visual_features": [], "suggested_locarno": [],
                        "needs_clarification": False}
    dq_fn = (design_query if callable(design_query)
             else (lambda *a, **k: design_query))
    st = mock.patch("sources.design.design_pipeline.design_query"
                    ".l0_product_from_text", new=dq_fn)
    sp = mock.patch("sources.design.design_pipeline.design_query"
                    ".parse_l0_json", new=dq_fn)
    ss = mock.patch("sources.design.design_pipeline.design_search"
                    ".search_designs", new=search)
    ps = []
    if pdf is not None:
        ps.append(mock.patch("sources.design.design_pipeline.design_image"
                             ".fetch_design_pdf", new=pdf))
    if judge is not None:
        ps.append(mock.patch("sources.design.design_pipeline.design_judge"
                             ".judge_product", new=judge))
    st.start(); sp.start(); ss.start()
    for p in ps:
        p.start()
    return st, sp, ss, ps


def _close(*handles):
    for group in reversed(handles[-1]):
        group.stop()
    handles[2].stop(); handles[1].stop(); handles[0].stop()


def test_l0_clarification_raises():
    ctx = _Ctx()
    h = _patch({"en_name": "Toy", "keywords": [], "visual_features": [],
                "suggested_locarno": [], "needs_clarification": True},
               search=_search([], limited=False))
    try:
        with pytest.raises(DesignNeedClarification):
            asyncio.run(run(PARAMS, ctx))
    finally:
        _close(*h)


def test_full_chain_success_digest_report_progress():
    ctx = _Ctx()
    h = _patch({"en_name": "Toy", "keywords": ["toy"], "visual_features": [],
                "suggested_locarno": [], "needs_clarification": False},
               search=_search([_cand()]),
               pdf=_ok_pdf, judge=_judge_ok)
    try:
        out = asyncio.run(run(PARAMS, ctx))
    finally:
        _close(*h)
    assert "USD1A" in out["digest"]["result_ids"]
    assert "非法律意见" in out["report_md"]
    assert out["digest"]["totals"]["high"] == 1
    assert len(ctx.progress_msgs) >= 4


def test_l1_rate_limited_raises_and_message():
    ctx = _Ctx()
    h = _patch({"en_name": "Toy", "keywords": ["toy"], "visual_features": [],
                "suggested_locarno": [], "needs_clarification": False},
               search=_search([], limited=True),
               pdf=_ok_pdf, judge=_judge_ok)
    try:
        with pytest.raises(DesignRuntimeError):
            asyncio.run(run(PARAMS, ctx))
    finally:
        _close(*h)
    assert any("暂不可用" in m for m in ctx.progress_msgs)


def test_l1_zero_hits_zero_contract():
    ctx = _Ctx()
    h = _patch({"en_name": "Toy", "keywords": ["toy"], "visual_features": [],
                "suggested_locarno": [], "needs_clarification": False},
               search=_search([]), pdf=_ok_pdf, judge=_judge_ok)
    try:
        out = asyncio.run(run(PARAMS, ctx))
    finally:
        _close(*h)
    assert "未检出" in out["report_md"]
    assert out["digest"]["result_ids"] == []


def test_l2_all_fail_text_fallback():
    ctx = _Ctx()
    h = _patch({"en_name": "Toy", "keywords": ["toy"], "visual_features": [],
                "suggested_locarno": [], "needs_clarification": False},
               search=_search([_cand()]),
               pdf=_raise_pdf, judge=_judge_ok)
    try:
        out = asyncio.run(run(PARAMS, ctx))
    finally:
        _close(*h)
    assert "文本维度候选" in out["report_md"] or any(
        "降级" in m or "文本维度" in m for m in ctx.progress_msgs)
    assert "非法律意见" in out["report_md"]


def test_l3_all_fail_notes_unmatched():
    ctx = _Ctx()
    h = _patch({"en_name": "Toy", "keywords": ["toy"], "visual_features": [],
                "suggested_locarno": [], "needs_clarification": False},
               search=_search([_cand()]),
               pdf=_ok_pdf, judge=_judge_empty)
    try:
        out = asyncio.run(run(PARAMS, ctx))
    finally:
        _close(*h)
    assert "未完成视觉比对" in out["report_md"] or any(
        "未完成视觉比对" in m for m in ctx.progress_msgs)
