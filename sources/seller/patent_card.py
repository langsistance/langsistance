"""Seller patent card assembly: claims text -> plain-language card.

Pure orchestration with an injectable provider (``complete_json``).
LLM failures degrade to ``llm_available:false`` instead of raising —
seller queries are free-tier high-frequency; a polite empty card beats a 5xx.
"""

import time

from sources.logger import Logger

logger = Logger("backend.log")

DISCLAIMER = "基于公开数据库自动分析，不保证检索穷尽，不构成法律意见"

# System copy fixed for the seller voice (spec §3.2-7): never surface
# claim-language terms; speak in 保护什么 / 撞不撞 / 能不能卖.
_CARD_SYSTEM = (
    "你是 CopiioAI 卖家专利安全台的专利解读助手，面向完全不懂专利法的跨境卖家。\n"
    "把下面的专利权利要求原文翻译成中文人话，只输出 JSON：\n"
    '{"protection_summary": "一句话说明这个专利保护什么（禁止出现"权利要求/本领域技术人员/实施例"等词）", '
    '"risk_level": "high|mid|low|expired", '
    '"next_step": "给卖家的下一步建议：可上架 / 需规避（提示改哪里） / 建议询价授权 / 已过期可参考"}'
)

_cache: dict[str, dict] = {}  # key -> {"expires_at": float, "card": dict}
_DEFAULT_TTL = 86400


def card_cache_get(key: str):
    entry = _cache.get(key)
    if not entry:
        return None
    if time.time() > entry["expires_at"]:
        _cache.pop(key, None)
        return None
    return entry["card"]


def card_cache_put(key: str, card: dict, ttl_seconds: int = _DEFAULT_TTL) -> None:
    _cache[key] = {"expires_at": time.time() + ttl_seconds, "card": card}


def _empty_card(patent_id: str, source: str) -> dict:
    return {
        "patent_id": patent_id,
        "source": source,
        "legal_status": None,
        "status_note": "状态核验中（M1.5）",
        "protection_summary": None,
        "risk_level": None,
        "next_step": None,
        "llm_available": False,
        "disclaimer": DISCLAIMER,
    }


async def build_patent_card(provider, claims_text: str, source: str,
                            patent_id: str, lang: str = "zh") -> dict:
    key = f"{source}:{patent_id}:{lang}"
    cached = card_cache_get(key)
    if cached is not None:
        return {"success": True, "card": cached, "cached": True}

    card = _empty_card(patent_id, source)
    try:
        parsed = await provider.complete_json(
            _CARD_SYSTEM,
            f"专利号：{patent_id}\n权利要求原文（摘录前 6000 字符）：\n{claims_text[:6000]}",
            max_retries=1,
        )
    except Exception as exc:  # noqa: BLE001 — degrade, never raise
        logger.error(f"seller card llm failed — {patent_id}: {exc}")
        parsed = None

    if isinstance(parsed, dict) and parsed.get("protection_summary"):
        card.update({
            "protection_summary": str(parsed["protection_summary"])[:500],
            "risk_level": parsed.get("risk_level")
            if parsed.get("risk_level") in ("high", "mid", "low", "expired") else None,
            "next_step": str(parsed.get("next_step") or "")[:500] or None,
            "llm_available": True,
        })
    card_cache_put(key, card)
    return {"success": True, "card": card, "cached": False}
