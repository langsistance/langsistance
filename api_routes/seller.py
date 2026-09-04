#!/usr/bin/env python3
"""Seller patent safety workbench routes (M1a slice).

POST /seller/patent_card — patent number -> plain-language card.

Thin orchestration only: classification and card assembly live in
sources/seller/* (unit-testable, framework-free).  Claims documents are
fetched through the existing patent_detail pipeline (_fetch_claims);
upstream misses degrade to 200 + success:false (Cloudflare swaps origin
5xx for its own CORS-less error page — same rationale as patent_detail).
"""

from fastapi import APIRouter, HTTPException, Request

from sources.logger import Logger
from sources.user.passport import verify_firebase_token
from sources.seller.query_classifier import classify_seller_query
from sources.seller.patent_card import build_patent_card

logger = Logger("backend.log")


def register_seller_routes(logger, config, provider):
    """Register seller workbench routes with dependency injection."""
    router = APIRouter()

    async def _fetch_claims_lazy(source: str, patent_id: str) -> dict:
        # Lazy import mirrors patent_detail's own internal function and
        # keeps module import graphs acyclic.
        from api_routes.patent_detail import _fetch_claims
        return await _fetch_claims(source, patent_id)

    @router.post("/seller/patent_card")
    async def seller_patent_card(http_request: Request):
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user["uid"])

        try:
            body = await http_request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        query = str(body.get("query") or "").strip()
        lang = body.get("lang") if body.get("lang") in ("zh", "en") else "zh"
        if not query:
            raise HTTPException(status_code=400, detail="query is required")

        classification = classify_seller_query(query)
        if classification["kind"] != "patent":
            return {"success": True, "kind": "product",
                    "message": "该输入走查一查检索（/seller/search，M1b）"}

        source = classification["source"]
        patent_id = classification["patent_id"]
        try:
            payload = await _fetch_claims_lazy(source, patent_id)
        except Exception as exc:  # noqa: BLE001 — degrade per module contract
            logger.error(f"seller card claims fetch failed — {patent_id}: {exc}")
            return {"success": True, "claims_available": False,
                    "message": "专利文本暂不可用，请稍后重试"}

        if not payload.get("success"):
            return {"success": True, "claims_available": False,
                    "message": payload.get("message") or "专利文本暂不可用"}

        claims = payload.get("claims") or []
        claims_text = "\n".join(
            str(c.get("text") or "") for c in claims if c.get("text")
        )
        if not claims_text.strip():
            return {"success": True, "claims_available": False,
                    "message": "专利文本为空，暂无法生成解读"}

        result = await build_patent_card(provider, claims_text, source,
                                         patent_id, lang=lang)
        result.update({"kind": "patent", "claims_available": True,
                       "user_id": user_id})
        return result

    return router
