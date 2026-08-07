"""China patent data client backed by Google Patents.

Provides the same interface as the original ``SipopClient`` so it can be
passed directly to ``fetch_examination_data()`` in ``china_examination.py``.

All patent-document data (claims, description, bibliographic info,
legal-status timeline) comes from patents.google.com HTML scraping.
Reexamination / invalidation decision documents are not available via
Google Patents; ``query_patent_review`` always returns an empty list.

Usage::

    client = ChinaPatentClient(google_client=google_instance)
    client.set_pub_number("CN107041743B")
    claims = await client.query_full_text(...)
    timeline = await client.query_legal_state_timeline(...)
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from sources.logger import Logger

if TYPE_CHECKING:
    from sources.google_patents_client import GooglePatentsClient

_logger = Logger("china_patent_client.log")


# ── Error types ─────────────────────────────────────────────────────────────────


class ChinaPatentError(Exception):
    """Base error for China patent data client failures."""


class ChinaPatentBackendError(ChinaPatentError):
    """Required client is not available."""


# ── Client ──────────────────────────────────────────────────────────────────────


class ChinaPatentClient:
    """Google-Patents-backed data client for China patent examination analysis.

    All public methods share the same signature as the original
    ``SipopClient`` so this can be dropped into the existing pipeline
    without changes to ``china_examination.py``.

    Call ``set_pub_number()`` before any query — the CN publication number
    comes from EPO family resolution (``family_context["cn_pub_number"]``).
    """

    def __init__(self, google_client: "GooglePatentsClient") -> None:
        self._google = google_client
        self._cn_pub_number: str = ""
        _logger.info("ChinaPatentClient initialized — Google Patents backend")

    # ── Patent context ───────────────────────────────────────────────────────

    def set_pub_number(self, pub_number: str) -> None:
        """Set the CN publication number for Google Patents queries.

        Format examples: ``"CN107041743B"``, ``"CN201080053868A"``.
        """
        self._cn_pub_number = pub_number

    # ── SipopClient-compatible interface ─────────────────────────────────────

    async def query_patent_review(
        self, application_doc_num: str, country: str = "CN",
    ) -> list[dict[str, Any]]:
        """Google Patents does not provide reexamination/invalidation decisions.

        Always returns an empty list.  The downstream pipeline handles this
        gracefully — it still produces a report with claims, description,
        and legal-status timeline.
        """
        return []

    async def query_law_state(
        self, application_doc_num: str,
    ) -> dict[str, Any]:
        """Query legal-status summary from Google Patents (IFI status text)."""
        if not self._cn_pub_number:
            return {}
        status = await self._google.query_legal_status_text(self._cn_pub_number)
        return {"date": "", "lawStatus": status, "lawStatusCode": ""}

    async def query_full_text(
        self, application_doc_num: str,
    ) -> dict[str, Any]:
        """Query claims + description from Google Patents.

        Returns a dict with ``claim`` (list[str]) and ``description``
        (list[str]).
        """
        if not self._cn_pub_number:
            _logger.warning(
                "china_patent: query_full_text — no pub_number set"
            )
            return {"claim": [], "description": [], "descriptionFigure": []}

        claims = await self._google.query_claims(self._cn_pub_number)
        description = await self._google.query_description(self._cn_pub_number)

        _logger.info(
            f"china_patent: query_full_text — "
            f"pub={self._cn_pub_number}, "
            f"claims={len(claims)}, desc_paras={len(description)}"
        )
        return {
            "claim": claims,
            "description": description,
            "descriptionFigure": [],
        }

    async def query_basic_info(
        self, application_doc_num: str,
    ) -> dict[str, Any]:
        """Query bibliographic info from Google Patents."""
        if not self._cn_pub_number:
            _logger.warning(
                "china_patent: query_basic_info — no pub_number set"
            )
            return {}
        return await self._google.query_basic_info(self._cn_pub_number)

    async def query_legal_state_timeline(
        self, application_doc_num: str, country: str = "CN",
    ) -> list[dict[str, Any]]:
        """Query legal-status event timeline from Google Patents.

        Returns a list of dicts with SIPOP-compatible keys:
        ``date``, ``lawStatusCode``, ``lawStatus``, ``lawStatusDetail``.
        """
        if not self._cn_pub_number:
            return []

        try:
            events = await self._google.query_legal_events(self._cn_pub_number)
        except Exception as exc:
            _logger.warning(
                f"china_patent: query_legal_state_timeline failed for "
                f"{self._cn_pub_number}: {exc}"
            )
            return []

        result: list[dict[str, Any]] = []
        for evt in events:
            result.append({
                "date": evt.get("date", ""),
                "lawStatusCode": evt.get("code", ""),
                "lawStatus": evt.get("title", ""),
                "lawStatusDetail": "",
            })
        _logger.info(
            f"china_patent: query_legal_state_timeline — "
            f"pub={self._cn_pub_number}, count={len(result)}"
        )
        return result

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def cn_pub_number(self) -> str:
        return self._cn_pub_number
