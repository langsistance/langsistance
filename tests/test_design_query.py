"""design_query L0 提示词与 JSON 解析 (design P1 T2)。

规格: docs/superpowers/specs/2026-09-06-us-design-clearance-design.md; task-2-brief.md。
- parse_l0_json: 剥 ```json 围栏→缺键补默认、坏 JSON 容错降级。
- l0_product_from_text: 文本降级入口, en_name=输入截 60, keywords=[原词]。
- L0_PROMPT_ZH: 通用视觉翻译提示, 零产品词固化。
"""
from sources.design.design_query import (
    parse_l0_json,
    l0_product_from_text,
    L0_PROMPT_ZH,
)


def test_parse_ok():
    raw = ('{"en_name":"Toy snake","keywords":["toy snake","robot"],'
           '"visual_features":["segmented"],"suggested_locarno":["21-01"],'
           '"needs_clarification":false}')
    p = parse_l0_json(raw)
    assert p["en_name"] == "Toy snake"
    assert p["needs_clarification"] is False


def test_parse_tolerates_markdown_fence_and_junk():
    fenced = ('```json\n{"en_name":"x","keywords":[],"visual_features":[],'
              '"suggested_locarno":[],"needs_clarification":false}\n```')
    p = parse_l0_json(fenced)
    assert p["en_name"] == "x"
    assert parse_l0_json("not json at all")["needs_clarification"] is True


def test_parse_fills_missing_keys_with_defaults():
    p = parse_l0_json('{"en_name":"y"}')
    assert p["en_name"] == "y"
    assert p["keywords"] == []
    assert p["visual_features"] == []
    assert p["suggested_locarno"] == []
    assert p["needs_clarification"] is False


def test_bad_json_yields_needs_clarification_empty_structure():
    p = parse_l0_json("garbage[][")
    assert p["needs_clarification"] is True
    assert p["en_name"] == ""
    assert p["keywords"] == []


def test_text_fallback_uses_first_meaningful_token():
    p = l0_product_from_text("帮我查遥控玩具蛇外观")
    assert p["en_name"]
    assert isinstance(p["keywords"], list)


def test_text_fallback_truncates_en_name_to_60():
    long_text = "x" * 120
    p = l0_product_from_text(long_text)
    assert len(p["en_name"]) <= 60


def test_prompt_is_generic():
    assert "toy snake" not in L0_PROMPT_ZH.lower()
    assert "输出 JSON" in L0_PROMPT_ZH
