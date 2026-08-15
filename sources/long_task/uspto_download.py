"""USPTO specification download shared by the long-task pipeline and the
ReAct chat loop.  Extracted verbatim from celery_worker.py so both paths
use the same implementation; behavior is byte-identical to the original.
"""

import os
from typing import Any

from sources.logger import Logger

from sources.long_task.text_extractor import (
    extract_text_from_binary,
    get_download_url_from_doc,
)

_get_download_url_from_doc = get_download_url_from_doc
_extract_text_from_binary = extract_text_from_binary

_DEFAULT_LOGGER = Logger("uspto_download.log")


def _module_log(msg: str) -> None:
    _DEFAULT_LOGGER.info(msg)


def _module_warn(msg: str) -> None:
    _DEFAULT_LOGGER.warning(msg)


def normalize_app_number(patent_id: str) -> str:
    """Strip commas, slashes, non-digits; return '' when too short."""
    app_number = (patent_id or "").strip().replace(",", "").replace("/", "")
    app_number = "".join(c for c in app_number if c.isdigit())
    if len(app_number) < 8:
        return ""
    return app_number


async def download_uspto_patent_text(
    patent_id: str,
    spec_selector_provider=None,
    logger: Any = None,
) -> tuple[str | None, bytes | None]:
    """Download USPTO specification text directly (two-step).

    Step 1: GET /api/v1/patent/applications/{appNumber}/documents
    Step 2: collect SPEC docs, LLM may pick preferred, download all,
            concatenate extracted text.

    Returns (text, binary):
      - (text, None)          — text extracted successfully
      - (None, binary_bytes)  — all specs failed text extraction, binary cached
      - (None, None)          — download failed entirely

    Never raises — all failures degrade to (None, None).
    """

    def _log(msg: str) -> None:
        (logger or _DEFAULT_LOGGER).info(msg)

    def _warn(msg: str) -> None:
        (logger or _DEFAULT_LOGGER).warning(msg)

    import asyncio
    import json as _json

    try:
        from sources.http_outbound import outbound_http

        # Normalize: strip commas, slashes, non-digits (US app numbers are pure digits)
        app_number = normalize_app_number(patent_id)
        if not app_number:
            _warn(f"[download] uspto_invalid_app_number — patent_id={patent_id}")
            return (None, None)
        headers = {'Accept': 'application/json'}
        uspto_key = os.getenv('USPTO_API_KEY', '')
        if uspto_key:
            headers['X-API-Key'] = uspto_key

        # Step 1: Get document list
        doc_list_url = (
            f"https://api.uspto.gov/api/v1/patent/applications/"
            f"{app_number}/documents"
        )
        _log(f"[download] uspto_step1 — url={doc_list_url}")
        resp = await _uspto_get_with_retry(doc_list_url, headers, timeout=20)
        if resp.status_code != 200:
            _warn(
                f"[download] uspto_step1_failed — status={resp.status_code}"
            )
            return (None, None)

        doc_list = resp.json() if resp.text else {}
        documents = (
            doc_list.get('documentBag', [])
            if isinstance(doc_list, dict)
            else []
        )
        # Keep a compact one-line summary only
        if documents:
            spec_count = sum(
                1 for d in documents
                if isinstance(d, dict) and d.get('documentCode') == 'SPEC'
            )
            _log(
                f"[download] uspto_step1_done — doc_count={len(documents)}, "
                f"spec_count={spec_count}"
            )
        if not documents:
            _warn(
                f"[download] uspto_no_documents — patent_id={app_number}"
            )
            return (None, None)

        # Step 2: Collect ALL specification documents (there may be multiple)
        spec_docs: list[dict] = []

        # Find all SPEC documents heuristically first (fast, no LLM)
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            code = str(doc.get('documentCode', '') or doc.get('documentTypeCode', ''))
            desc = str(doc.get('documentCodeDescriptionText', '') or doc.get('documentTypeName', ''))
            if 'SPEC' in code.upper() or 'specification' in desc.lower():
                spec_docs.append(doc)

        # If the LLM gave us a preferred index, move that one to the front
        if spec_selector_provider and len(documents) > 1:
            doc_lines = []
            for i, doc in enumerate(documents):
                if not isinstance(doc, dict):
                    continue
                doc_lines.append(_json.dumps({
                    'index': i,
                    'code': doc.get('documentCode') or doc.get('documentTypeCode', ''),
                    'description': doc.get('documentCodeDescriptionText') or doc.get('documentTypeName', ''),
                    'pageCount': doc.get('pageTotalQuantity') or doc.get('pageCount', ''),
                    'hasDownload': bool(doc.get('downloadOptionBag')),
                }, ensure_ascii=False))
            try:
                selection = await spec_selector_provider.complete_json(
                    "You are a patent document classifier. From a list of USPTO "
                    "patent application documents, identify the specification "
                    "(说明书). The specification is typically:\n"
                    "- code = 'SPEC' or description containing 'Specification'\n"
                    "- The main detailed description of the invention\n"
                    "- NOT the abstract, claims-only sequence listing, or drawings\n"
                    "- NOT administrative documents like Power of Attorney, "
                    "Fee Payment, Notice of Allowance\n\n"
                    "Return JSON: {\"selected_index\": <index of specification>, "
                    "\"reason\": \"<brief explanation>\"}",
                    f"Patent application: {app_number}\n"
                    f"Available documents:\n" + "\n".join(doc_lines),
                )
                if selection and isinstance(selection, dict):
                    idx = selection.get('selected_index')
                    if isinstance(idx, int) and 0 <= idx < len(documents):
                        preferred = documents[idx]
                        _log(
                            f"[download] llm_selected_spec — index={idx}, "
                            f"code={preferred.get('documentCode')}, "
                            f"reason={selection.get('reason', '')[:100]}"
                        )
                        # Move preferred to front (deduplicate if already in list)
                        spec_docs = [preferred] + [d for d in spec_docs if d is not preferred]
            except Exception as e:
                _warn(
                    f"[download] llm_spec_selection_failed: {e}"
                )

        if not spec_docs:
            _warn(
                f"[download] uspto_no_spec_found — patent_id={app_number}"
            )
            return (None, None)

        _log(
            f"[download] uspto_spec_candidates — count={len(spec_docs)}, "
            f"indices={[documents.index(d) for d in spec_docs]}"
        )

        # Step 3: Download ALL SPEC documents and concatenate their text.
        # A single patent application may have multiple SPEC files; downloading
        # all of them gives the most complete specification.
        all_parts: list[str] = []
        first_binary_fallback: bytes | None = None
        for attempt, spec_doc in enumerate(spec_docs):
            spec_code = spec_doc.get('documentCode') or spec_doc.get('documentTypeCode', '?')

            spec_url = get_download_url_from_doc(spec_doc)
            format_label = _guess_format_from_url(spec_url)
            _log(
                f"[download] uspto_spec[{attempt+1}/{len(spec_docs)}] — "
                f"code={spec_code}, format={format_label}, "
                f"url={spec_url[:100]}"
            )

            text, binary = await _download_uspto_spec_with_redirect(
                spec_doc, app_number, headers,
            )
            chars = len(text.strip()) if text else 0
            if chars > 200:
                _log(
                    f"[download] uspto_spec[{attempt+1}] ok — "
                    f"format={format_label}, chars={chars}"
                )
                all_parts.append(text.strip())
            else:
                _log(
                    f"[download] uspto_spec[{attempt+1}] skipped "
                    f"({chars} chars)"
                )
                if binary is not None and first_binary_fallback is None:
                    first_binary_fallback = binary
                    _log(
                        f"[download] uspto_spec[{attempt+1}] binary_cached "
                        f"for vision fallback — len={len(binary)}"
                    )

        if all_parts:
            combined = "\n\n".join(all_parts)
            _log(
                f"[download] uspto_all_specs_done — "
                f"parts={len(all_parts)}, total_chars={len(combined)}"
            )
            return (combined, None)

        if first_binary_fallback is not None:
            _log(
                f"[download] uspto_all_text_failed_but_binary_cached — "
                f"patent_id={app_number}, binary_len={len(first_binary_fallback)}"
            )
            return (None, first_binary_fallback)

        _warn(
            f"[download] uspto_all_specs_failed — patent_id={app_number}, "
            f"tried={len(spec_docs)}"
        )
        return (None, None)
    except Exception as e:
        _warn(
            f"[download] uspto_direct_error — patent_id={patent_id}, error={e}"
        )
        return (None, None)


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


async def _uspto_get_with_retry(url: str, headers: dict, timeout: int = 30):
    """GET *url* via outbound_http.

    Retry on 400 / 429 is handled centrally by OutboundHttpClient for all
    USPTO API URLs.  This wrapper keeps the calling code unchanged.
    """
    import asyncio as _asyncio
    from sources.http_outbound import outbound_http

    return await _asyncio.to_thread(
        outbound_http.get, url, purpose="patent_download",
        headers=headers, timeout=timeout,
    )


async def _download_uspto_spec_with_redirect(
    spec_doc: dict,
    app_number: str,
    headers: dict,
) -> tuple[str | None, bytes | None]:
    """Download USPTO specification, following redirect URLs if needed.

    USPTO download URLs may return a text/JSON body containing another URL
    (e.g. "Please use redirect URL: https://...").  We follow at most one
    redirect.  Pattern taken from uspto_download.py.

    Returns (text, binary):
      - (text, None)          — text extracted successfully
      - (None, binary_bytes)  — binary downloaded but text extraction failed
      - (None, None)          — download failed entirely
    """

    import asyncio

    from sources.http_outbound import outbound_http

    spec_url = _get_download_url_from_doc(spec_doc)
    if not spec_url:
        _module_warn(
            f"[download] uspto_spec_no_url — app={app_number}, "
            f"spec_doc_keys={list(spec_doc.keys()) if spec_doc else 'N/A'}"
        )
        return (None, None)
    for hop in range(2):  # max 1 redirect
        resp = await _uspto_get_with_retry(spec_url, headers, timeout=30)
        if resp.status_code != 200:
            _module_warn(
                f"[download] uspto_spec_hop{hop}_failed — status={resp.status_code}"
            )
            return (None, None)

        content_type = resp.headers.get('Content-Type', '').lower()
        content = resp.text or ''

        # xmlarchive URLs always deliver tar binaries regardless of Content-Type
        # (USPTO may label them application/xml even though they are tar files).
        force_binary = 'xmlarchive' in spec_url.lower()

        # If response looks like a file (not text/JSON), extract text properly
        if force_binary or (content_type and not any(t in content_type for t in ('text/', 'json', 'xml', 'html'))):
            _module_log(
                f"[download] uspto_spec_binary — type={content_type}, "
                f"len={len(resp.content)}"
            )
            extracted = _extract_text_from_binary(
                resp.content, content_type, spec_url,
                skip_pdf_extraction=True,
            )
            if extracted and len(extracted) > 100:
                return (extracted, None)
            _module_warn(
                f"[download] uspto_spec_extract_empty — "
                f"type={content_type}, len={len(resp.content)}"
            )
            return (None, resp.content)

        # Check if the text response contains a redirect URL
        stripped = content.strip()
        if not stripped:
            _module_warn(
                f"[download] uspto_spec_empty — app={app_number}"
            )
            return (None, None)

        # Try to extract a redirect URL from the response
        from sources.dynamic_tool_params import _extract_first_url
        redirect_url = _extract_first_url(stripped)
        if redirect_url and redirect_url != spec_url:
            _module_log(
                f"[download] uspto_spec_redirect — to={redirect_url[:120]}"
            )
            spec_url = redirect_url
            continue

        # No redirect: this IS the content
        _module_log(
            f"[download] uspto_spec_done — len={len(stripped)}"
        )
        return (stripped, None)

    return (None, None)
