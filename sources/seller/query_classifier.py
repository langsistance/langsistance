"""Query classification for the seller patent card flow.

Pure logic — no FastAPI / DB imports so it stays unit-testable.

Adaptation note (2026-09-04, verified against sources/patent_number_parser.py):
- ``decide_number_source`` returns ``'cn' | 'uspto' | None`` (medium-confidence
  candidates return None), so this module falls back to the top candidate's
  ``country`` when the hard-signal routing is undecided.
- Candidate dicts carry ``display / raw / country / confidence / lookups``
  (no ``pub`` / ``digits`` keys); ``patent_id`` uses ``display`` (canonical
  form the detail pipeline accepts) and ``matched`` is the original
  punctuated span from the user text (parser ``raw``/``display`` strip
  punctuation, which would break UI echo-back).
"""

import re

from sources.patent_number_parser import (
    parse_patent_identifiers,
    decide_number_source,
)

_MAX_LEN = 200

# Mirrors the parser's input shapes (prefixed granted/application numbers,
# D-design numbers, bare US application serials) but preserves the original
# punctuation/spacing so the matched span can be echoed back verbatim.
_RAW_SPAN_RE = re.compile(
    r"(?i)\b(?:US|CN)\s?(?:D\s?)?[0-9][0-9,\./ ]{3,}[0-9]"
    r"|[0-9]{2}/[0-9,]{3,}"
)


def _matched_span(text: str) -> str:
    """First patent-number-shaped span of *text*, punctuation preserved."""
    m = _RAW_SPAN_RE.search(text)
    return m.group(0).strip() if m else text.strip()[:40]


def classify_seller_query(text: str) -> dict:
    """Classify a seller workbench query into patent-number or product intent.

    Returns {"kind": "patent", "source": "uspto"|"baiten", "patent_id",
    "matched"} when the text carries a recognizable US/CN patent number,
    else {"kind": "product"}.  Never raises.
    """
    if not text or not text.strip():
        return {"kind": "product"}
    stripped = text.strip()
    if len(stripped) > _MAX_LEN:
        return {"kind": "product"}

    candidates = parse_patent_identifiers(stripped)
    if not candidates:
        return {"kind": "product"}

    source = decide_number_source(candidates)
    if source is None:
        # Hard-signal routing undecided (medium confidence, e.g. bare US
        # application serials): route by the top candidate's country.
        country = next(
            (c.get("country") for c in candidates if c.get("country") in ("US", "CN")),
            None,
        )
        if country is None:
            return {"kind": "product"}
        source = "baiten" if country == "CN" else "uspto"
    elif source == "cn":
        source = "baiten"

    top = candidates[0]
    return {
        "kind": "patent",
        "source": source,
        "patent_id": str(top.get("display") or _matched_span(stripped)),
        "matched": _matched_span(stripped),
    }
