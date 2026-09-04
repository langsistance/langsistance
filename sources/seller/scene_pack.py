"""Seller scene profile pack (A1).

One place that owns the seller *voice*: the system-prompt addendum applied
when a request carries ``scene="seller"``.  Pure constants + helpers — no
imports beyond stdlib so tests stay trivial.

Red-line (spec §3.2-7): the copy below must never surface patent-law jargon
to the seller; it also must not contain any single test/query wording
(generic phrasing only — "测试提问词不固化").
"""

SELLER_SCENE_NAME = "卖家查专利"          # scenes 表行名（种子 SQL 另行落地）
SELLER_SCENE_ID = 2                       # 与种子 SQL 保持一致（id=1 为美国专利检索）

_SELLER_VOICE_ZH = (
    "你是 CopiioAI 卖家专利安全台，当前提问来自跨境卖家场景。\n"
    "1. 全程说中文人话，禁用任何专利法术语（概念一律改用日常说法，如\"它保护什么、"
    "和你的产品撞不撞、到期没、能不能卖\"）；面向卖家的文字不得出现面向专利"
    "从业者的行话。\n"
    "2. 检索类问题优先给结论与专利号；给不出结论时明说范围（查了什么、没查到什么），"
    "不编造。\n"
    "3. 输出一律带一句\"分析供参考，不构成法律意见\"。"
)

_SELLER_VOICE_EN = (
    "You are CopiioAI Seller Patent Safety Desk; this query comes from a "
    "cross-border seller. Use plain language, never patent-law jargon; "
    "lead with conclusions and patent numbers; state search scope when "
    "nothing is found; end with 'not legal advice'."
)

_SCENE_IDS = {"seller": SELLER_SCENE_ID, "pro": 1}


def seller_voice_addendum(lang: str = "zh") -> str:
    """System-prompt addendum for the seller scene ('' if lang unsupported)."""
    if lang == "zh":
        return _SELLER_VOICE_ZH
    if lang == "en":
        return _SELLER_VOICE_EN
    return ""


def scene_id_for_request(scene: str | None) -> int | None:
    """Map a request-level scene token to a scenes-table id (None = default)."""
    if not scene:
        return None
    return _SCENE_IDS.get(scene)
