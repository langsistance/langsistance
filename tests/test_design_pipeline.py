"""test_design_pipeline: design_pipeline.run 编排 (design P1 T7 + review I3 vision L0)。

6 场景全 mock (禁真实网络)。说明: L1/L2 的网络 fetch 由模块级 mock 取代;
流程①-⑥ 逐一驱动 run 以验证 digest/report/进度与异常分流。
"""
import asyncio
from unittest import mock

import pytest

from sources.design.design_pipeline import (DesignNeedClarification,
                                            DesignRuntimeError, run)
from sources.design.design_risk import DesignCandidate, JudgeVerdict
from sources.design.design_vision import DesignVisionError

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


# ── review I3: 纯图入口 (product_text 空 + 图) 走视觉 L0, 非恒 clarify ──
#
# Gating: product_image_refs 在文件系统解析出 base64 且 product_text 空 →
# `_vision_resolve` → design_vision.call_vision(唯一 product data-uri, L0_PROMPT_ZH)
# → parse_l0_json。`_to_product_images` 打桩回固定 data-uri 以驱动纯图分支
# (filesystem 无关); call_vision / L1 / L2 / L3 全 mock 禁真网。

_VISION_URI = "data:image/png;base64,AAABCAaa=="


def _patch_vision_l0(call_vision):
    """装配纯图 L0 桩: _to_product_images→固定 data-uri; call_vision→给定异步函数;
    L1/L2/L3 全 stub。返回 handles 列表。"""
    import sources.design.design_pipeline as dp
    hs = [
        mock.patch.object(dp, "_to_product_images", return_value=[_VISION_URI]),
        mock.patch.object(dp.design_vision, "call_vision", new=call_vision),
        mock.patch.object(dp.design_search, "search_designs",
                          new=_search([_cand()])),
        mock.patch.object(dp.design_image, "fetch_design_pdf", new=_ok_pdf),
        mock.patch.object(dp.design_judge, "judge_product", new=_judge_ok),
    ]
    for h in hs:
        h.start()
    return hs


def _vision_params():
    return {"product_image_refs": ["real.png"], "product_text": "",
            "source": "us_design"}


def test_pure_image_vision_l0_proceeds_full_chain():
    ctx = _Ctx()

    async def _ok_vision(images_base64, prompt, post=None, timeout=90, *,
                         config=None):
        # 纯图 L0 视觉成功: 回含 en_name 的 L0 JSON → 后续 L1 命中可 proceed
        assert config is None or config.get("enabled")  # 不 assert 具体
        return '{"en_name": "pirate ship mug", "keywords": ["mug"], ' \
               '"visual_features": [], "suggested_locarno": []}'

    h = _patch_vision_l0(_ok_vision)
    try:
        out = asyncio.run(run(_vision_params(), ctx))
    finally:
        for ph in h:
            ph.stop()
    assert "USD1A" in out["digest"]["result_ids"]
    assert "非法律意见" in out["report_md"]


def test_pure_image_vision_l0_failure_still_clarifies():
    ctx = _Ctx()

    async def _fail_vision(images_base64, prompt, post=None, timeout=90, *,
                           config=None):
        raise DesignVisionError("vision disabled")

    h = _patch_vision_l0(_fail_vision)
    try:
        with pytest.raises(DesignNeedClarification):
            asyncio.run(run(_vision_params(), ctx))
    finally:
        for ph in h:
            ph.stop()


# ── 2026-09-07 修复: 图+文本时视觉 L0 为准, 请求语不再当产品名检索 ──────
# 线上: 传图 + "帮我看这个产品有没有外观专利风险" → product_text 非空走文本
# 纯函数 → 整句中文当 en_name → L1 空检零命中 (1.6s 完成)。修复后: 有图 →
# 视觉 L0 (文本仅作参考并入提示); 视觉拿不到且文本像英文品名才兜底文本。

def _patch_text_vision_l0(call_vision, search=None):
    """装配"图+文本"L0 桩: _to_product_images→data-uri; call_vision→给定;
    search 捕获参数; L2/L3 stub。返回 (handles, captured)。"""
    import sources.design.design_pipeline as dp
    captured = {}
    if search is None:
        async def _search(*a, **k):
            captured["en_name"] = a[1] if len(a) > 1 else None
            return {"candidates": [_cand()], "rate_limited": False}
        search = _search
    hs = [
        mock.patch.object(dp, "_to_product_images", return_value=[_VISION_URI]),
        mock.patch.object(dp.design_vision, "call_vision", new=call_vision),
        mock.patch.object(dp.design_search, "search_designs", new=search),
        mock.patch.object(dp.design_image, "fetch_design_pdf", new=_ok_pdf),
        mock.patch.object(dp.design_judge, "judge_product", new=_judge_ok),
    ]
    for h in hs:
        h.start()
    return hs, captured


def test_text_image_vision_l0_wins_over_request_sentence():
    """请求语 + 图: 视觉 en_name 驱动检索, 中文请求语不进 L1。"""
    ctx = _Ctx()
    seen_prompt = {}

    async def _ok_vision(images_base64, prompt, post=None, timeout=90, *,
                         config=None):
        seen_prompt["prompt"] = prompt
        return ('{"en_name": "insulated mug", "keywords": ["mug", "cup"], '
                '"visual_features": [], "suggested_locarno": []}')

    h, captured = _patch_text_vision_l0(_ok_vision)
    try:
        out = asyncio.run(run(
            {"product_image_refs": ["p.png"],
             "product_text": "帮我看这个产品有没有外观专利风险",
             "source": "us_design"}, ctx))
    finally:
        for ph in h:
            ph.stop()
    # 视觉被调用, 提示含用户文本作参考
    assert "帮我看这个产品" in seen_prompt["prompt"]
    # L1 用的是视觉 en_name, 不是请求语
    assert captured.get("en_name") == "insulated mug"
    assert "USD1A" in out["digest"]["result_ids"]


def test_text_image_vision_clarify_cjk_text_still_clarifies():
    """视觉说需澄清 + 文本仅中文请求语(无英文品名) → 仍澄清而非垃圾检索。"""
    ctx = _Ctx()

    async def _clarify_vision(images_base64, prompt, post=None, timeout=90, *,
                              config=None):
        return ('{"en_name": "", "keywords": [], "visual_features": [], '
                '"suggested_locarno": [], "needs_clarification": true}')

    h, _ = _patch_text_vision_l0(_clarify_vision)
    try:
        with pytest.raises(DesignNeedClarification):
            asyncio.run(run(
                {"product_image_refs": ["p.png"],
                 "product_text": "帮我看这个产品有没有外观专利风险",
                 "source": "us_design"}, ctx))
    finally:
        for ph in h:
            ph.stop()


def test_text_image_vision_fails_english_name_falls_back_to_text():
    """视觉不可用 + 文本首段是英文品名 → 兜底文本画像继续检索。"""
    ctx = _Ctx()

    async def _fail_vision(images_base64, prompt, post=None, timeout=90, *,
                           config=None):
        raise DesignVisionError("vision down")

    h, captured = _patch_text_vision_l0(_fail_vision)
    try:
        out = asyncio.run(run(
            {"product_image_refs": ["p.png"],
             "product_text": "Insulated travel mug 500ml",
             "source": "us_design"}, ctx))
    finally:
        for ph in h:
            ph.stop()
    assert captured.get("en_name") == "Insulated travel mug 500ml"
    assert "USD1A" in out["digest"]["result_ids"]
