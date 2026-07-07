"""USPTO prosecution document classification and download orchestration.

Determines which documents from the USPTO document list are relevant for
prosecution history analysis and downloads their text content.

The download functions accept a fetch callback to avoid circular imports
with celery_worker.py (which provides rate-limited USPTO HTTP functions).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from sources.dynamic_tool_params import _extract_first_url
from sources.logger import Logger
from sources.long_task.text_extractor import (
    extract_text_from_binary,
    get_download_url_from_doc,
)

_logger = Logger("prosecution_downloader.log")


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class ProsecutionDoc:
    """A single prosecution document with metadata and extracted text."""

    category: str  # 'office_action' | 'applicant_response' | 'amendment' | ...
    priority: int  # 1=must, 2=recommended, 3=skip
    document_code: str
    description: str
    page_count: int = 0
    download_url: str = ""
    text: str = ""  # populated after download
    binary: bytes | None = None  # raw bytes for vision/OCR fallback
    file_format: str = ""  # 'DOCX' | 'PDF' | 'XML' | 'UNKNOWN' — set after download
    raw_doc: dict | None = None  # original USPTO documentBag entry


@dataclass
class ProsecutionDocManifest:
    """Categorized list of prosecution documents from USPTO document list."""

    must_download: list[ProsecutionDoc] = field(default_factory=list)
    recommended: list[ProsecutionDoc] = field(default_factory=list)
    skipped: list[ProsecutionDoc] = field(default_factory=list)
    total_in_bag: int = 0

    @property
    def all_to_download(self) -> list[ProsecutionDoc]:
        return self.must_download + self.recommended

    @property
    def download_count(self) -> int:
        return len(self.must_download) + len(self.recommended)


# ── Priority classification rules ─────────────────────────────────────────────
#
# Each rule dict has:
#   codes:       set of uppercase document codes to match
#   descriptions: list of lowercase substrings to match in description text
#
# Matching: a document matches if its documentCode is in codes OR its
# documentCodeDescriptionText contains any of the description substrings
# (case-insensitive).

PRIORITY_1_RULES: dict[str, dict] = {
    "office_action": {
        "codes": {"CTNF", "CTFR", "CTAV", "CTRS", "CTEQ"},
        "descriptions": [
            "non-final office action",
            "final office action",
            "office action",
            "restriction requirement",
            "non-final rejection",
            "final rejection",
            "ex parte quayle",
            "advisory action",
            "examiner's action",
        ],
    },
    "applicant_response": {
        "codes": set(),
        "descriptions": [
            "response to office action",
            "applicant arguments",
            "response after final",
            "after final response",
            "response to restriction",
            "response to election",
            "amendments and arguments",
            "remarks",
            "remarks/arguments",
            "arguments/remarks",
            "response to non-final",
            "response to final",
        ],
    },
    "amendment": {
        "codes": {"CLM", "WCLM"},  # Claims & Withdrawn Claims — essential for amendment tracking
        "descriptions": [
            "claims",
            "amendment",
            "preliminary amendment",
            "after final amendment",
            "amendment after final",
            "amendment under",
            "supplemental amendment",
            "amended",
        ],
    },
    "notice_of_allowance": {
        "codes": {"NOA"},
        "descriptions": [
            "notice of allowance",
            "notice of allowability",
            "issue notification",
        ],
    },
}

PRIORITY_2_RULES: dict[str, dict] = {
    "specification": {
        "codes": {"SPEC"},
        "descriptions": [
            "specification",
            "spec",
        ],
    },
    "ids": {
        "codes": {"IDS"},
        "descriptions": [
            "information disclosure statement",
            "information disclosure",
            "disclosure statement",
        ],
    },
    "interview_summary": {
        "codes": {"EXIN", "INTVW"},
        "descriptions": [
            "interview summary",
            "examiner interview",
            "interview agenda",
            "interview record",
            "intervie",
        ],
    },
    "appeal": {
        "codes": set(),
        "descriptions": [
            "appeal brief",
            "appeal decision",
            "reply brief",
            "examiner's answer",
            "notice of appeal",
            "appeal",
        ],
    },
    "rce": {
        "codes": {"RCE"},
        "descriptions": [
            "request for continued examination",
            "continued examination",
        ],
    },
    "petition": {
        "codes": set(),
        "descriptions": [
            "petition",
        ],
    },
}

# Priority 3: Administrative documents — no AI analysis value.
# Matched by description substrings (case-insensitive).
PRIORITY_3_SKIP_DESCRIPTIONS: list[str] = [
    "issue fee",
    "filing receipt",
    "receipt",
    "power of attorney",
    "application data sheet",
    "drawing",
    "small entity",
    "fee worksheet",
    "assignment",
    "address change",
    "oath or declaration",
    "declaration",
    "oath",
    "notice to file missing parts",
    "acceptance",
    "notice of incomplete",
    "electronic acknowledgement",
    "fee transmittal",
    "micro entity",
    "change of correspondence",
    "certificate",
    "patent term adjustment",
    "status",
    "request for refund",
    "notice of publication",
    "certified copy",
    "sequence listing",
    "notice of abandonment",
    "express abandonment",
    "corrected",
    "change of address",
    "correspondence address",
    "entity status",
    "maintenance fee",
    "extension of time",
    "request for prioritized",
    "track one",
    "preliminary amendment",  # pre-filing amendment, less valuable
    "authorization",
    "transmittal",
    "abstract",
    "bibliographic",
    "letter",
    "fee payment",
    "issue notification",
    "notice of informal",
    "notice to file corrected",
    " applicant ",
]

# Additional document CODES that are never useful for prosecution analysis.
# These are administrative forms, bibliographic sheets, etc.
PRIORITY_3_SKIP_CODES: set[str] = {
    "BIB",         # Bibliographic data sheet
    "LET",         # Transmittal letter
    "N417.PYMT",   # Fee payment form
    "IIFW",        # Issue fee worksheet
    "SRFW",        # Fee worksheet
    "SRNT",        # Notice/transmittal
    "1449",        # Form 1449
    "892",         # Notice of references (listed in OA anyway)
    "FOR",         # Foreign reference (listed in OA/IDS)
    "REF.OTHER",   # Other references (listed in IDS)
    "ABST",        # Abstract
    "DRW",         # Drawings
    "PTO.1449",    # Form PTO-1449
    "P.PAMPHLET",  # Patent pamphlet (reference)
    "P.N.101.CONV",# Conversion form
    "371P",        # Form 371
    "SCORE",       # Search scorecard
}


# ── Classification ─────────────────────────────────────────────────────────────


def _matches_any_description(description: str, substrings: list[str]) -> bool:
    """Check if description contains any of the given substrings (case-insensitive)."""
    desc_lower = description.lower()
    return any(sub in desc_lower for sub in substrings)


def _classify_single_document(doc: dict) -> ProsecutionDoc | None:
    """Classify a single USPTO documentBag entry.

    Returns a ProsecutionDoc with category and priority, or None if
    the document doesn't have enough metadata.
    """
    code = str(doc.get("documentCode", "") or doc.get("documentTypeCode", "")).strip()
    desc = str(
        doc.get("documentCodeDescriptionText", "")
        or doc.get("documentTypeName", "")
    ).strip()

    if not code and not desc:
        return None

    # Extract download URL
    download_url = ""
    download_bag = doc.get("downloadOptionBag", [])
    if isinstance(download_bag, list) and download_bag:
        first_option = download_bag[0]
        if isinstance(first_option, dict):
            download_url = first_option.get("downloadUrl", "") or first_option.get(
                "url", ""
            )

    page_count = int(
        doc.get("pageTotalQuantity", 0) or doc.get("pageCount", 0) or 0
    )

    # Check priority 1 rules
    for category, rules in PRIORITY_1_RULES.items():
        codes = rules.get("codes", set())
        descriptions = rules.get("descriptions", [])
        if code.upper() in codes or _matches_any_description(desc, descriptions):
            # Note: "Preliminary Amendment" appears in BOTH Priority 1 amendment
            # rules AND Priority 3 skip list.  Priority 1 is checked first, so
            # pre-filing amendments are correctly classified as useful.
            return ProsecutionDoc(
                category=category,
                priority=1,
                document_code=code,
                description=desc,
                page_count=page_count,
                download_url=download_url,
                raw_doc=doc,
            )

    # Check priority 2 rules
    for category, rules in PRIORITY_2_RULES.items():
        codes = rules.get("codes", set())
        descriptions = rules.get("descriptions", [])
        if code.upper() in codes or _matches_any_description(desc, descriptions):
            return ProsecutionDoc(
                category=category,
                priority=2,
                document_code=code,
                description=desc,
                page_count=page_count,
                download_url=download_url,
                raw_doc=doc,
            )

    # Check priority 3 skip rules (by description OR by code)
    if _matches_any_description(desc, PRIORITY_3_SKIP_DESCRIPTIONS) or code.upper() in PRIORITY_3_SKIP_CODES:
        return ProsecutionDoc(
            category="administrative",
            priority=3,
            document_code=code,
            description=desc,
            page_count=page_count,
            download_url=download_url,
            raw_doc=doc,
        )

    # Everything else: default to priority 2 (potentially useful, let AI decide)
    return ProsecutionDoc(
        category="other",
        priority=2,
        document_code=code,
        description=desc,
        page_count=page_count,
        download_url=download_url,
        raw_doc=doc,
    )


def classify_prosecution_documents(document_bag: list[dict]) -> ProsecutionDocManifest:
    """Classify all documents in a USPTO documentBag into priority categories.

    Args:
        document_bag: List of document dicts from USPTO API response
                      (each has documentCode, documentCodeDescriptionText, etc.).

    Returns:
        ProsecutionDocManifest with must_download, recommended, and skipped lists.
    """
    manifest = ProsecutionDocManifest(total_in_bag=len(document_bag))

    for doc in document_bag:
        if not isinstance(doc, dict):
            continue

        classified = _classify_single_document(doc)
        if classified is None:
            continue

        if classified.priority == 1:
            manifest.must_download.append(classified)
        elif classified.priority == 2:
            manifest.recommended.append(classified)
        else:
            manifest.skipped.append(classified)

    _logger.info(
        f"[prosecution] classified documents — "
        f"total={manifest.total_in_bag}, "
        f"must_download={len(manifest.must_download)}, "
        f"recommended={len(manifest.recommended)}, "
        f"skipped={len(manifest.skipped)}"
    )

    # Log what we're downloading vs skipping
    for doc in manifest.must_download:
        _logger.info(
            f"[prosecution] PRIORITY_1 — category={doc.category}, "
            f"code={doc.document_code}, desc={doc.description[:80]}"
        )
    for doc in manifest.recommended:
        _logger.info(
            f"[prosecution] PRIORITY_2 — category={doc.category}, "
            f"code={doc.document_code}, desc={doc.description[:80]}"
        )

    return manifest


# ── Document grouping ─────────────────────────────────────────────────────────


def group_documents_by_category(
    downloaded: list[ProsecutionDoc],
) -> dict[str, list[ProsecutionDoc]]:
    """Group downloaded documents by their category for AI analysis.

    Returns dict like:
        {
            'office_actions': [...],
            'applicant_responses': [...],
            'amendments': [...],
            'notice_of_allowance': [...],
            'ids': [...],
            'interviews': [...],
            ...
        }
    """
    groups: dict[str, list[ProsecutionDoc]] = {}
    for doc in downloaded:
        if doc.category not in groups:
            groups[doc.category] = []
        groups[doc.category].append(doc)
    return groups


# ── Download helpers ──────────────────────────────────────────────────────────


# Type alias for the fetch function: (url, headers, timeout) -> response-like object
FetchFunc = Callable[[str, dict, int], Awaitable[Any]]


def _guess_format_from_url(url: str) -> str:
    """Guess the file format from a download URL."""
    url_lower = url.lower()
    if url_lower.endswith('.docx') or 'ms_word' in url_lower:
        return 'DOCX'
    if url_lower.endswith('.xml') or 'xmlarchive' in url_lower:
        return 'XML'
    if url_lower.endswith('.pdf'):
        return 'PDF'
    return 'UNKNOWN'


async def download_single_document(
    doc: ProsecutionDoc,
    fetch_func: FetchFunc,
    app_number: str,
    headers: dict,
) -> ProsecutionDoc:
    """Download and extract text from a single prosecution document.

    Returns:
        The same ProsecutionDoc with .text, .binary, and .file_format populated.
    """
    if not doc.download_url and doc.raw_doc:
        doc.download_url = get_download_url_from_doc(doc.raw_doc) or ""

    if not doc.download_url:
        _logger.warning(
            f"[prosecution] result=no_url — code={doc.document_code}, "
            f"desc={doc.description[:60]}"
        )
        return doc

    url = doc.download_url
    doc.file_format = _guess_format_from_url(url)

    try:
        for _hop in range(2):
            resp = await fetch_func(url, headers, 30)
            if resp.status_code != 200:
                _logger.warning(
                    f"[prosecution] result=http_{resp.status_code} — "
                    f"fmt={doc.file_format}, category={doc.category}, "
                    f"code={doc.document_code}, desc={doc.description[:50]}"
                )
                return doc

            content_type = resp.headers.get("Content-Type", "").lower()

            # Binary response — extract text
            if content_type and not any(
                t in content_type for t in ("text/", "json", "xml", "html")
            ):
                # If Content-Type suggests a different format than URL, update
                if not doc.file_format or doc.file_format == 'UNKNOWN':
                    if 'pdf' in content_type:
                        doc.file_format = 'PDF'
                    elif 'word' in content_type or 'msword' in content_type:
                        doc.file_format = 'DOCX'

                extracted = extract_text_from_binary(
                    resp.content, content_type, url, skip_pdf_extraction=False
                )
                if extracted and len(extracted.strip()) > 50:
                    doc.text = extracted.strip()
                    _logger.info(
                        f"[prosecution] result=ok — "
                        f"fmt={doc.file_format}, category={doc.category}, "
                        f"code={doc.document_code}, chars={len(doc.text)}, "
                        f"desc={doc.description[:40]}"
                    )
                    return doc

                # Text extraction failed (likely scanned PDF)
                doc.binary = resp.content
                _logger.warning(
                    f"[prosecution] result=no_text — "
                    f"fmt={doc.file_format}, category={doc.category}, "
                    f"code={doc.document_code}, "
                    f"binary_size={len(resp.content)}, "
                    f"desc={doc.description[:40]}"
                )
                return doc

            # Text-like response — check for redirect
            content = resp.text or ""
            redirect_url = _extract_first_url(content.strip())
            if redirect_url and redirect_url != url:
                _logger.info(
                    f"[prosecution] redirect — "
                    f"from={url[:60]} -> to={redirect_url[:60]}"
                )
                url = redirect_url
                continue

            # Text response IS the content
            if len(content.strip()) > 50:
                doc.text = content.strip()
                if not doc.file_format or doc.file_format == 'UNKNOWN':
                    doc.file_format = 'TEXT'
                _logger.info(
                    f"[prosecution] result=ok_text — "
                    f"fmt={doc.file_format}, category={doc.category}, "
                    f"code={doc.document_code}, chars={len(doc.text)}, "
                    f"desc={doc.description[:40]}"
                )
                return doc

            # Content too short
            _logger.warning(
                f"[prosecution] result=empty — "
                f"fmt={doc.file_format}, category={doc.category}, "
                f"code={doc.document_code}, len={len(content.strip())}, "
                f"desc={doc.description[:40]}"
            )
            return doc

    except Exception as e:
        _logger.warning(
            f"[prosecution] result=error — "
            f"fmt={doc.file_format}, category={doc.category}, "
            f"code={doc.document_code}, "
            f"error={type(e).__name__}: {e}"
        )
        return doc

    return doc


async def download_prosecution_documents(
    docs: list[ProsecutionDoc],
    fetch_func: FetchFunc,
    app_number: str,
    headers: dict | None = None,
) -> list[ProsecutionDoc]:
    """Download text for a list of classified prosecution documents.

    Downloads each document sequentially (respecting USPTO rate limits)
    and logs a detailed summary by result type and format.
    """
    if headers is None:
        headers = {"Accept": "application/json"}
        uspto_key = os.getenv("USPTO_API_KEY", "")
        if uspto_key:
            headers["X-API-Key"] = uspto_key

    for i, doc in enumerate(docs):
        await download_single_document(doc, fetch_func, app_number, headers)

    # ── Detailed summary ──
    ok = [d for d in docs if d.text]
    no_text = [d for d in docs if not d.text and d.binary]
    http_fail = [d for d in docs if not d.text and not d.binary and d.download_url]
    no_url = [d for d in docs if not d.download_url]

    _logger.info(
        f"[prosecution] download_summary — "
        f"total={len(docs)}, "
        f"ok={len(ok)}, "
        f"no_text_likely_scanned={len(no_text)}, "
        f"http_error={len(http_fail)}, "
        f"no_url={len(no_url)}"
    )

    # Per-format breakdown
    from collections import Counter
    ok_fmts = Counter(d.file_format for d in ok)
    fail_fmts = Counter(d.file_format for d in no_text)
    _logger.info(
        f"[prosecution] format_ok — {dict(ok_fmts)}"
    )
    _logger.info(
        f"[prosecution] format_no_text — {dict(fail_fmts)}"
    )

    # List the 5 most valuable failures for diagnostics
    if no_text:
        _logger.info(
            f"[prosecution] top_no_text_samples — " +
            " | ".join(
                f"fmt={d.file_format},cat={d.category},code={d.document_code},desc={d.description[:40]}"
                for d in no_text[:5]
            )
        )

    return docs
