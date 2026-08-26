"""Patent source (uspto / cnipa / auto) detection from text signals.

The keyword tables here are shared by the API layer
(``api_routes/core.py:_detect_patent_source``) and the agent layer
(``general_agent.create_agent``) so the built-in dual-source search tool
registration agrees with the long-task routing.  Pure functions only.
"""


def detect_patent_source_text(
    query: str, conv_history: list = None,
) -> str:
    """Detect patent source (uspto/cnipa/auto) from query + history text.

    Order (keywords take priority — what the user is searching):
      1. Explicit patent-office keywords (uspto / 美国专利 / cnipa / 中国专利)
      2. Company-name inference (US companies → uspto, CN companies → cnipa)
      3. Default: ``auto`` (no signal — caller decides dual/single)

    Same keyword tables and precedence as the long-task detector; keep
    them in sync when extending.
    """
    conv_history = conv_history or []
    combined = f"{query or ''} " + " ".join(
        str(m.get("content", "")) for m in conv_history
        if isinstance(m, dict)
    )
    combined_lower = combined.lower()

    # ── 1. Explicit patent-office keywords (primary — user intent) ──
    uspto_keywords = [
        "uspto", "美国专利", "美国专利商标局", "united states patent",
        "us patent", "us application",
    ]
    cnipa_keywords = [
        "cnipa", "中国专利", "中国国家知识产权", "国家知识产权局",
        "chinese patent", "china patent", "zldsj",
    ]
    if any(kw in combined_lower for kw in uspto_keywords):
        return "uspto"
    if any(kw in combined_lower for kw in cnipa_keywords):
        return "cnipa"

    # ── 2. Company names (lower priority than explicit keywords) ──
    cn_company_keywords = [
        "华为", "小米", "oppo", "vivo", "腾讯", "阿里巴巴", "百度",
        "比亚迪", "宁德时代", "中兴", "大疆", "字节跳动", "中芯国际",
        "京东方", "格力", "美的", "海尔", "联想", "蔚来", "小鹏", "理想",
        "寒武纪", "地平线", "紫光", "长江存储", "长鑫",
    ]
    us_company_keywords = [
        "apple", "google", "microsoft", "tesla", "intel", "amd",
        "nvidia", "qualcomm", "ibm", "meta", "amazon", "broadcom",
        "micron", "cisco", "oracle", "hp", "dell",
    ]
    if any(kw in combined_lower for kw in us_company_keywords):
        return "uspto"
    if any(kw in combined_lower for kw in cn_company_keywords):
        return "cnipa"

    # ── 3. No signal ──
    return "auto"


def map_source_for_tool_route(source: str) -> str:
    """Map a detected source to the tool-registration semantics.

    uspto → "uspto"   (现有 USPTO 场景工具，不注册内置检索工具)
    cnipa → "cn"      (佰腾语义：中国专利检索工具，不再触发 zldsj 路由)
    auto  → "dual"    (双源并行：综合专利检索工具)
    """
    if source == "cnipa":
        return "cn"
    if source == "uspto":
        return "uspto"
    return "dual"
