#!/usr/bin/env python3
"""Patent detail endpoints for the split-view results page.

GET /patent/{source}/{patent_id}/spec    — 说明书 PDF（内嵌阅读器代理地址）
GET /patent/{source}/{patent_id}/claims  — 权利要求列表

Documents come from USPTO's file-wrapper download (the same mechanism the
prosecution-history long task uses): resolve the application number, fetch
the PEDS document list and locate the SPEC / CLM documents.  The spec
endpoint returns the USPTO PDF behind the existing lazy-download proxy
(``/uspto/download``) so the frontend embeds the original PDF in a viewer
— the same way patent documents are displayed.  Claims are downloaded and
parsed into a structured list.  ``source`` is validated against the known
set but does not change the download path — USPTO is the single data
source.

Auth: Firebase bearer token (same pattern as other web routes).
"""

import re
from html import unescape

from fastapi import APIRouter, HTTPException, Request

from sources.logger import Logger
from sources.user.passport import verify_firebase_token

logger = Logger("backend.log")

VALID_SOURCES = {"uspto", "google_patents", "baiten"}

_CLAIM_START_PATTERN = re.compile(r"(?m)^\s*(\d{1,3})\.\s*")

# Dependent claims open by referencing an earlier claim — standard shapes:
# "The method of claim 1", "The apparatus according to claim 2",
# "A method according to any one of claims 1 to 3", "The method as
# claimed in claim 1", "The method of any preceding claim".
_DEPENDENT_OPENERS = re.compile(
    r"^(?:the|a)\s+[\w\s/’-]{2,60}?\s+"
    r"(?:of|according\s+to|as\s+claimed\s+in|as\s+defined\s+in|as\s+recited\s+in)\s+"
    r"(?:any\s+one\s+of\s+)?(?:the\s+)?(?:preceding|previous)?\s*claims?\b",
    re.IGNORECASE,
)

# Chinese dependent openers: 如权利要求1所述 / 根据权利要求1-3中任一项所述 …
_DEPENDENT_OPENERS_CN = re.compile(
    r"^(?:如|根据)权利要求(?:\d+(?:[-–—至到]\d+)?|前述|上述|任一|任一项)"
)

_XML_TAG_PATTERN = re.compile(r"<[^>]+>")

# Block-ish elements whose closing tags become line breaks when tags are
# stripped — keeps claim-number line starts intact for text fallback.
_BLOCK_CLOSE_PATTERN = re.compile(
    r"</(?:claim-text|claim|claims|amended-claim|p|paragraph|heading|"
    r"section|div)\s*>|<br\s*/?>",
    re.IGNORECASE,
)

# Amendment status markers prefixing claims in response/amendment documents:
# "(original) …", "(previously presented) …", "(canceled)" …
_AMENDMENT_STATUS_PATTERN = re.compile(
    r"^\s*\(?\s*(original|previously presented|new|currently amended|amended|"
    r"canceled|withdrawn)\s*\)?\s*[:.\-]?\s*",
    re.IGNORECASE,
)

# Document header/footer noise lines that pollute OCR/extracted claims text.
_NOISE_LINE_PATTERNS = [
    re.compile(r"^page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^serial\s+no\.?\s*:?\s*\S", re.IGNORECASE),
    re.compile(r"^response to office action\s*$", re.IGNORECASE),
    re.compile(r"^mailed on\s+\S", re.IGNORECASE),
    re.compile(r"^amendments? to the claims", re.IGNORECASE),
    re.compile(r"^the following is a complete listing", re.IGNORECASE),
    re.compile(r"^(what is claimed is|we claim|claims?)\s*:?\s*$", re.IGNORECASE),
    re.compile(r"^(application|filing|docket)\s+(number|date)\s*:?", re.IGNORECASE),
    re.compile(r"^-?\d+-$"),  # page numbers like "-2-"
    re.compile(r"^amdt\s+date\s", re.IGNORECASE),
    # OCR page-footer garbage in VASTEC CLM.XML, e.g.
    # "7C2B18MM\261532 Amendment to 2025-09-10 FOA 4937-"
    re.compile(r"^[A-Z0-9]{4,}\\\d+"),
    re.compile(r"^.*Amendment to \d{4}-\d{2}-\d{2} FOA", re.IGNORECASE),
]


class PatentDetailError(Exception):
    """Base error for patent detail fetch failures."""


def build_claims_payload(claims: list[str]) -> dict:
    """Build the claims response payload; independence follows the opener."""
    if not claims:
        return {"success": False, "claims": []}
    payload_claims = []
    for index, text in enumerate(claims, start=1):
        cleaned, status = _strip_claim_status_markers(text)
        payload_claims.append({
            "number": index,
            "text": cleaned,
            "status": status,
            "independent": status == "active" and _is_independent_claim(cleaned, index == 1),
        })
    return {"success": True, "claims": payload_claims}


def _is_independent_claim(claim_text: str, is_first: bool) -> bool:
    """True when the claim does not open by referencing an earlier claim."""
    if is_first:
        return True
    first_line = (claim_text or "").strip().splitlines()
    if not first_line:
        return True
    # Tolerate a leading claim-number prefix ("2. The method of claim 1…")
    line = re.sub(r"^\d{1,3}\.\s*", "", first_line[0].strip(), count=1)
    return not (_DEPENDENT_OPENERS.match(line) or _DEPENDENT_OPENERS_CN.match(line))


def _strip_claim_status_markers(claim_text: str) -> tuple[str, str]:
    """Strip a leading claim-number prefix and amendment status marker.

    Returns ``(cleaned_text, status)`` where status is ``"active"`` or
    ``"canceled"`` — amendment documents prefix claims with
    "(original) …" / "(previously presented) …" / "(canceled)".
    """
    text = re.sub(
        r"^\d{1,3}\.\s*", "", (claim_text or "").strip(), count=1
    )
    match = _AMENDMENT_STATUS_PATTERN.match(text)
    if not match:
        return text.strip(), "active"
    status = (
        "canceled"
        if match.group(1).lower() in ("canceled", "withdrawn")
        else "active"
    )
    return text[match.end():].strip(), status


def _strip_document_noise(text: str) -> str:
    """Remove header/footer noise lines from extracted document text."""
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept = [
        line
        for line in lines
        if not any(pattern.match(line.strip()) for pattern in _NOISE_LINE_PATTERNS)
    ]
    return "\n".join(kept)


def split_claims_text(text: str) -> list[str]:
    """Split raw claims-document text into individual claims.

    Each claim starts with its number at line start ("1. ...", "2. ...").
    Text before the first claim number is discarded (headers/footers).
    """
    if not text:
        return []
    matches = list(_CLAIM_START_PATTERN.finditer(text))
    if not matches:
        return []
    claims = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            claims.append(body)
    return claims


# Sentence-case articles open new claims ("A system …", "An apparatus …");
# lowercase continuations ("a data collection …", "an element …") do not.
_UNNUMBERED_CLAIM_STARTERS = re.compile(r"^(?:A|An)\s+\S")


def split_unnumbered_claims(text: str) -> list[str]:
    """Split an unnumbered claims section into claims by paragraph.

    Fallback for documents whose claims carry no "N. " numbers (Word
    auto-numbered lists lose their numbers in extraction).  A paragraph
    starts a new claim when it opens with "A/An <Noun>" or with a
    dependent claim opener ("The X of claim N …"); every other paragraph
    joins the previous claim as a continuation.
    """
    if re.search(r"\n\s*\n", text or ""):
        # PDF-style: paragraphs separated by blank lines
        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", text)
            if p.strip()
        ]
    else:
        # DOCX-style: one paragraph per line, single newlines
        paragraphs = [
            line.strip()
            for line in (text or "").splitlines()
            if line.strip()
        ]
    claims: list[str] = []
    for paragraph in paragraphs:
        first_line = paragraph.splitlines()[0].strip()
        stripped = re.sub(r"^\d{1,3}\.\s*", "", first_line, count=1)
        starts_new = bool(
            _UNNUMBERED_CLAIM_STARTERS.match(stripped)
            or _DEPENDENT_OPENERS.match(stripped)
            or _DEPENDENT_OPENERS_CN.match(stripped)
        )
        if not claims and not starts_new:
            # Preamble ("What is claimed is:", "We claim:") — skip until
            # the first claim actually starts.
            continue
        if not starts_new:
            claims[-1] = f"{claims[-1]}\n{paragraph}"
        else:
            claims.append(paragraph)
    return claims


def _strip_xml_tags(text: str) -> str:
    """Strip XML tags (SPEC.XML / CLM.XML payloads) and unescape entities.

    Closing tags of block-ish elements become newlines so line-start
    patterns (claim numbering) survive the strip; inline tags collapse
    to spaces.
    """
    if not text or "<" not in text:
        return text
    stripped = _BLOCK_CLOSE_PATTERN.sub("\n", text)
    stripped = _XML_TAG_PATTERN.sub(" ", stripped)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    return unescape(stripped).strip()


def _find_spec_document(document_bag: list) -> dict | None:
    """Locate the specification document in a USPTO documentBag."""
    if not isinstance(document_bag, list):
        return None
    for doc in document_bag:
        if not isinstance(doc, dict):
            continue
        code = str(doc.get("documentCode", "") or "").strip().upper()
        desc = str(doc.get("documentCodeDescriptionText", "") or "").lower()
        if code == "SPEC" or "specification" in desc:
            return doc
    return None


def _find_claims_document(document_bag: list) -> dict | None:
    """Locate the claims document in a USPTO documentBag."""
    if not isinstance(document_bag, list):
        return None
    for doc in document_bag:
        if not isinstance(doc, dict):
            continue
        code = str(doc.get("documentCode", "") or "").strip().upper()
        if code in ("CLM", "WCLM"):
            return doc
    for doc in document_bag:
        if not isinstance(doc, dict):
            continue
        desc = str(doc.get("documentCodeDescriptionText", "") or "").lower()
        if "claim" in desc:
            return doc
    return None


def _claim_number_from(element) -> int | None:
    """Extract a claim number — ``num`` attribute (classic) or a
    ``ClaimNumber`` child element (ST.96 VASTEC schema)."""
    num = str(element.get("num") or "").lstrip("0")
    if num.isdigit():
        return int(num)
    for child in element:
        if child.tag.rsplit("}", 1)[-1].lower() not in ("claimnumber", "claim-number"):
            continue
        num = "".join(child.itertext()).strip().lstrip("0")
        if num.isdigit():
            return int(num)
    return None


def _text_without_deletions(element) -> str:
    """Element text with <Del> (editorial deletion) content removed."""
    parts = []
    if element.text:
        parts.append(element.text)
    for child in element:
        if child.tag.rsplit("}", 1)[-1].lower() == "del":
            continue
        parts.append(_text_without_deletions(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


_DESIGN_CLAIM_PATTERN = re.compile(
    r"ornamental design|as shown and described", re.IGNORECASE
)
_PAGE_SUFFIX_PATTERN = re.compile(r"\s*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE)
# Design claims end with this standard phrase — everything after it is
# page-number/footer noise (OCR sometimes glues "Page 2" to the text).
_DESIGN_CLAIM_END_PATTERN = re.compile(
    r"as shown and described\s*[.;:]?", re.IGNORECASE
)


def _extract_design_claim(root) -> str | None:
    """Extract the single design-patent claim from a CLM.XML document.

    Design patents (D-numbers) carry no ``<Claim>`` elements — the claim
    is a plain paragraph mentioning the ornamental design, e.g.
    "The ornamental design for the False Eyelashes, as shown and
    described."  The text is cut at that standard closing phrase, so
    page markers and OCR remnants after it are dropped; editorial
    deletions are ignored.
    """
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1].lower() != "p":
            continue
        text = _text_without_deletions(element).strip()
        if not _DESIGN_CLAIM_PATTERN.search(text):
            continue
        end_match = _DESIGN_CLAIM_END_PATTERN.search(text)
        if end_match:
            text = text[: end_match.end()].rstrip()
            if text:
                return text
        text = _PAGE_SUFFIX_PATTERN.sub("", text).strip()
        if text:
            return text
    return None


def _parse_claims_xml(text: str) -> list[dict] | None:
    """Parse a USPTO CLM.xml payload into structured claims.

    Handles the claim schemas in the wild:
    - classic ``<claim num="…">`` with ``<claim-text>`` children;
    - ST.96 VASTEC ``<uspat:Claim>`` with a ``<pat:ClaimNumber>`` child
      and ``<uspat:ClaimText>`` segments (namespaces are matched by local
      name, and OCR footer segments are dropped);
    - design patents (D-numbers) that carry no claim elements at all —
      the single ornamental-design paragraph is extracted instead.

    Returns ``[{"number": int, "text": str}]`` in document order, or None
    when the payload is not claims XML (or no claims could be found) so
    callers can fall back to text parsing.
    """
    if not text:
        return None
    try:
        import xml.etree.ElementTree as _ET
        root = _ET.fromstring(text)
    except Exception:
        return None

    claims: list[dict] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1].lower() != "claim":
            continue
        parts = []
        for child in element:
            if child.tag.rsplit("}", 1)[-1].lower() not in ("claimtext", "claim-text"):
                continue
            part = "".join(child.itertext()).strip()
            if not part:
                continue
            if any(pattern.match(part) for pattern in _NOISE_LINE_PATTERNS):
                continue
            parts.append(part)
        claim_text = " ".join(parts).strip()
        if not claim_text:
            continue
        claims.append({
            "number": _claim_number_from(element) or len(claims) + 1,
            "text": claim_text,
        })

    if not claims:
        design_claim = _extract_design_claim(root)
        if design_claim:
            return [{"number": 1, "text": design_claim}]

    return claims or None


def _get_baiten_client():
    """Build a BaitenClient from config; PatentDetailError when unset."""
    from sources.baiten_client import BaitenClient
    from sources.long_task.config import get_baiten_config

    cfg = get_baiten_config()
    if not cfg["app_key"] or not cfg["app_secret"]:
        raise PatentDetailError("Baiten not configured (BAITEN_APP_KEY/APP_SECRET)")
    return BaitenClient(cfg["app_key"], cfg["app_secret"], cfg["gateway_url"])


def _extract_doc_field(doc: dict, *keys) -> str:
    """First non-empty string value among *keys* in *doc* (flat or in data)."""
    data = doc.get("data") if isinstance(doc.get("data"), dict) else doc
    for container in (doc, data):
        if not isinstance(container, dict):
            continue
        for key in keys:
            value = str(container.get(key) or "").strip()
            if value:
                return value
    return ""


def _normalize_baiten_date(value: str) -> str:
    """YYYY-MM-DD / YYYYMMDD / YYYY.MM.DD → YYYYMMDD (getFile contract)."""
    if not value:
        return ""
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits[:8]


def _build_baiten_download_url(pub_num: str, pub_date: str) -> str:
    return (f"/baiten/download?pub_num={pub_num}"
            f"&pub_date={_normalize_baiten_date(pub_date)}")


def _flatten_baiten_claims(body: dict) -> list[str]:
    """Flatten Baiten patentClaimses[] into claim texts.

    Rows are {claim, claimsNum, claimsParentNum}; only the claim text is
    needed here — independence is re-derived from the opener by the
    existing build_claims_payload rules (如权利要求1所述 …).
    """
    data = body.get("data") if isinstance(body.get("data"), dict) else body
    rows = None
    for container in (data, body):
        if isinstance(container, dict):
            rows = container.get("patentClaimses")
            if rows:
                break
    texts: list[str] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        text = str(row.get("claim") or "").strip()
        if text and text not in texts:
            texts.append(text)
    return texts


async def _fetch_baiten_spec(patent_id: str) -> dict:
    """Resolve a CN patent spec PDF proxy URL via Baiten.

    getDoc fills pubDate (needed by the file API), then the PDF is
    streamed through ``/baiten/download`` so the frontend inline viewer
    keeps working unchanged.  (getSpec text is deferred: the spec tab
    renders pdf_url only, and a text channel would need frontend work.)
    """
    from sources.baiten_client import BaitenError

    try:
        client = _get_baiten_client()
        doc = await client.get_doc(patent_id)
    except BaitenError as exc:
        raise PatentDetailError(str(exc)) from exc
    pub_date = _normalize_baiten_date(_extract_doc_field(doc, "pd", "pubDate"))
    if not pub_date:
        raise PatentDetailError(
            f"No publication date for Baiten patent {patent_id}")
    return {"success": True,
            "pdf_url": _build_baiten_download_url(patent_id, pub_date)}


def _looks_like_baiten_app_num(value: str) -> bool:
    """True when *value* is a CN application number (ends with the check
    digit ``.X``) rather than a publication number (ends with a kind-code
    letter like A/B/U/S).
    """
    return bool(re.search(r"\.\d$", (value or "").strip()))


async def _fetch_baiten_claims(patent_id: str) -> dict:
    """Fetch structured CN claims via Baiten.

    The frontend now sends the application number when it has one, so the
    broken extService getDoc hop (signature gate passes but the data
    service reports system error, 2026-08-27) is skipped entirely.
    Publication numbers still resolve through getDoc.  Claims are fetched
    AUTH then APP and flattened into the existing {number, text, status,
    independent} payload.  Scanned-only patents fall back to the PDF
    proxy URL (publication number only).
    """
    from sources.baiten_client import BaitenError

    client = _get_baiten_client()
    if _looks_like_baiten_app_num(patent_id):
        app_num = patent_id
        doc = None
    else:
        try:
            doc = await client.get_doc(patent_id)
        except BaitenError as exc:
            raise PatentDetailError(str(exc)) from exc
        app_num = _extract_doc_field(doc, "an", "appNum", "applicationNumber")
        if not app_num:
            raise PatentDetailError(
                f"No application number for Baiten patent {patent_id}")

    claim_texts: list[str] = []
    for pat_type in ("AUTH", "APP"):
        try:
            body = await client.get_claims(app_num, pat_type)
        except BaitenError:
            continue
        claim_texts = _flatten_baiten_claims(body)
        if claim_texts:
            break

    if claim_texts:
        return build_claims_payload(claim_texts)

    # No structured claims (scanned original) — inline viewer fallback.
    # The download proxy needs the publication number, which the app-num
    # path does not carry; honest degrade instead of a wrong URL.
    if doc is not None:
        pub_date = _normalize_baiten_date(
            _extract_doc_field(doc, "pd", "pubDate"))
        if pub_date:
            return {"success": True,
                    "pdf_url": _build_baiten_download_url(
                        patent_id, pub_date)}
    raise PatentDetailError(
        f"No claims available for Baiten patent {patent_id}")


async def _fetch_spec_pdf(source: str, patent_id: str) -> dict:
    """Resolve the specification PDF and return its proxy URL.

    The URL points at the lazy-download proxy (``/uspto/download`` for
    USPTO, ``/baiten/download`` for Baiten CN) so the frontend can embed
    the original PDF in an inline viewer without exposing the upstream
    API key — the same mechanism patent document rows already use.
    """
    if source == "baiten":
        return await _fetch_baiten_spec(patent_id)
    from sources import uspto_download
    from sources import uspto_download
    from sources.dynamic_tool_params import _build_uspto_download_proxy_url
    from sources.long_task.text_extractor import (
        USPTO_PDF_PREFERRED_MIME_ORDER,
        get_download_url_from_doc,
    )

    try:
        app_number = await uspto_download.resolve_application_number(patent_id)
    except ValueError as exc:
        raise PatentDetailError(str(exc)) from exc
    document_bag = await uspto_download.fetch_document_bag(app_number)
    if document_bag is None:
        raise PatentDetailError(
            f"USPTO document list unavailable for application {app_number}"
        )
    spec_doc = _find_spec_document(document_bag)
    if spec_doc is None:
        raise PatentDetailError(
            f"No specification document in USPTO file wrapper for {app_number}"
        )
    pdf_url = get_download_url_from_doc(
        spec_doc, mime_order=USPTO_PDF_PREFERRED_MIME_ORDER
    )
    if not pdf_url:
        raise PatentDetailError(
            f"No downloadable specification PDF for application {app_number}"
        )
    return {"success": True, "pdf_url": _build_uspto_download_proxy_url(pdf_url)}


_XML_MIME_ORDER = ("XML", "application/xml", "text/xml")
_DOCX_MIME_ORDER = (
    "MS_WORD",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
)


async def _fetch_claims(source: str, patent_id: str) -> dict:
    """Fetch the claims document for *patent_id*.

    Baiten CN source: structured patentClaimses via getDoc+getClaims,
    with an AUTH→APP fallback and a PDF proxy fallback for scanned
    originals.  USPTO preference order:
    1. XML (xmlarchive CLM.XML) — structured parse into the claims list.
    2. DOCX — text-layer parse (numbered / paragraph fallback).
    3. PDF — no extraction at all: return the proxy URL and let the
       frontend show the original PDF in the inline viewer (scanned
       documents never get OCR'd).
    """
    if source == "baiten":
        return await _fetch_baiten_claims(patent_id)
    from sources import uspto_download
    from sources.dynamic_tool_params import _build_uspto_download_proxy_url
    from sources.long_task.text_extractor import (
        USPTO_PDF_PREFERRED_MIME_ORDER,
        get_download_url_from_doc,
    )

    try:
        app_number = await uspto_download.resolve_application_number(patent_id)
    except ValueError as exc:
        raise PatentDetailError(str(exc)) from exc
    document_bag = await uspto_download.fetch_document_bag(app_number)
    if document_bag is None:
        raise PatentDetailError(
            f"USPTO document list unavailable for application {app_number}"
        )
    claims_doc = _find_claims_document(document_bag)
    if claims_doc is None:
        raise PatentDetailError(
            f"No claims document in USPTO file wrapper for {app_number}"
        )

    # 1. XML — structured CLM.xml parse (numbers live in ClaimNumber).
    _bag_items = (document_bag if isinstance(document_bag, list)
                  else (document_bag.get('items', []) if isinstance(document_bag, dict) else []))
    if get_download_url_from_doc(
        claims_doc, mime_order=_XML_MIME_ORDER, fallback_to_any=False
    ):
        text = await uspto_download.download_document_text(
            claims_doc, mime_order=_XML_MIME_ORDER
        )
        if text:
            structured = _parse_claims_xml(text)
            if structured:
                return build_claims_payload([claim["text"] for claim in structured])
            logger.warning(
                f"claims xml parse found no claims — source={source}, "
                f"id={patent_id}, raw_head={text[:500]!r}"
            )

    # 2. DOCX — text-layer parse with numbered / paragraph fallbacks.
    if get_download_url_from_doc(
        claims_doc, mime_order=_DOCX_MIME_ORDER, fallback_to_any=False
    ):
        text = await uspto_download.download_document_text(
            claims_doc, mime_order=_DOCX_MIME_ORDER
        )
        if text:
            cleaned = _strip_document_noise(_strip_xml_tags(text))
            claims = split_claims_text(cleaned)
            if not claims:
                # Word auto-numbered lists lose their numbers in
                # extraction — split by paragraph with claim-starter
                # heuristics instead.
                claims = split_unnumbered_claims(cleaned)
            if claims:
                return build_claims_payload(claims)
            logger.warning(
                f"claims parse found no claims — source={source}, "
                f"id={patent_id}, chars={len(cleaned)}, head={cleaned[:300]!r}"
            )

    # 3. PDF — inline viewer, no extraction.  Strict: never substitute a
    # non-PDF format into the viewer.
    pdf_url = get_download_url_from_doc(
        claims_doc,
        mime_order=USPTO_PDF_PREFERRED_MIME_ORDER,
        fallback_to_any=False,
    )
    if not pdf_url:
        raise PatentDetailError(
            f"No downloadable claims document for application {app_number}"
        )
    logger.info(f"claims branch probe — app={app_number}: PDF fallback selected")
    return {"success": True, "pdf_url": _build_uspto_download_proxy_url(pdf_url)}


def register_patent_detail_routes(logger, config):
    """Register patent detail routes with dependency injection."""
    router = APIRouter()

    @router.get("/patent/{source}/{patent_id}/spec")
    async def patent_spec(source: str, patent_id: str, http_request: Request):
        verify_firebase_token(http_request.headers.get("Authorization"))
        if source not in VALID_SOURCES:
            raise HTTPException(status_code=400, detail="Unsupported source")
        if not patent_id or len(patent_id) > 40:
            raise HTTPException(status_code=400, detail="Invalid patent_id")
        try:
            payload = await _fetch_spec_pdf(source, patent_id)
        except PatentDetailError as exc:
            logger.error(f"spec fetch failed — source={source}, id={patent_id}: {exc}")
            # Expected upstream misses are data conditions, not server
            # errors: return 200 + success:false instead of a 5xx, because
            # Cloudflare swaps origin 5xx responses for its own error page
            # (which carries no CORS headers) and the browser then blocks
            # the response as a CORS failure.
            return {"success": False, "message": "Patent specification unavailable"}
        return {"success": True, **payload}

    @router.get("/patent/{source}/{patent_id}/claims")
    async def patent_claims(source: str, patent_id: str, http_request: Request):
        verify_firebase_token(http_request.headers.get("Authorization"))
        if source not in VALID_SOURCES:
            raise HTTPException(status_code=400, detail="Unsupported source")
        if not patent_id or len(patent_id) > 40:
            raise HTTPException(status_code=400, detail="Invalid patent_id")
        try:
            payload = await _fetch_claims(source, patent_id)
        except PatentDetailError as exc:
            logger.error(f"claims fetch failed — source={source}, id={patent_id}: {exc}")
            return {"success": False, "message": "Patent claims unavailable"}
        if not payload.get("success"):
            # Honest degrade: claims not parseable from the public record.
            # Same 200 + success:false contract — see the spec endpoint.
            return {"success": False, "message": "权利要求暂不可用，请通过 PDF 原文查看"}
        return payload

    return router
