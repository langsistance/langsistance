"""US patent identifier normalization.

Users paste identifiers in many shapes — bare digits, country prefix,
commas, kind codes (B2/A1/B1).  The naive "keep every digit" extraction
mangles them: US9019058B2 becomes "90190582" (the kind-code digit leaks
into the number), which then 404s/403s against the USPTO API.

This module parses the identifier shape instead and returns the bare
number, ready to search or use as an application number.
"""

import re

# Identifier shapes, group 1 = the number (digits, commas, slashes, dots),
# group 2 = the kind code (B2 / A1 / B / E1 / P3 ...):
#   US9019058B2 | US 9,019,058 B2 | 9019058 | 13850906
#   US20250103146A1 | 17/027,484 | US12345678B2
#   USRE45678E1 | USPP12345P3
_PATENT_ID_RE = re.compile(
    r"^(?:US|CN|EP|WO|JP)?\s*(?:RE|PP)?\s*"
    r"([0-9][0-9,/. ]*[0-9])\s*"
    r"([A-Z]{1,3}[0-9]?)?$",
    re.IGNORECASE,
)


def extract_us_patent_digits(raw: str | None) -> str:
    """Return the bare digits of a patent identifier, kind code removed.

    ``US9019058B2`` → ``"9019058"`` (not ``"90190582"``), ``17/027,484``
    → ``"17027484"``, ``US12345678B2`` → ``"12345678"``.  Inputs that do
    not match any known shape keep the legacy all-digits behavior so
    previously-working callers are unaffected.  Never raises.
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    match = _PATENT_ID_RE.match(text)
    if match:
        return re.sub(r"[^0-9]", "", match.group(1))
    return "".join(ch for ch in text if ch.isdigit())


def kind_code_of(raw: str | None) -> str:
    """Return the kind code of an identifier (``"B2"`` for
    ``US9019058B2``), or ``""`` when the input carries none.

    Callers that require a specific identifier type (e.g. the long-task
    submit endpoint wants an 8-digit application number) use this to
    reject grant/publication numbers — "US12000123B2" must not pass an
    8-digit check as if it were the application "12000123".
    """
    text = str(raw or "").strip()
    if not text:
        return ""
    match = _PATENT_ID_RE.match(text)
    return match.group(2).upper() if match and match.group(2) else ""
