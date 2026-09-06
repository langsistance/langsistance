"""US design-clearance upload-intent detector (design P1 T7b).

Appearance/侵权询检 plug point for the API core route.  A product-appearance
clearance review is meaningful only with an uploaded product IMAGE (the design
pipeline's L2/L3 visual comparison is the report hinge; L1 word search is just the
candidate width over it).  The domain cue words below are generic *capability*
vocabulary (appearance-design review), not a single test question — no user-query
synonym is baked into code or prompt text.  Pure + zero IO so it is unit-traceable
off-server without the heavy api_routes import chain.
"""

_DESIGN_CLEARANCE_CUES = (
    "外观设计", "外观专利", "外观侵权", "设计专利", "侵权比对", "外观查询",
)


def has_design_cue(query: str) -> bool:
    """True when *query* mentions an appearance-design review domain cue."""
    text = query or ""
    return any(cue in text for cue in _DESIGN_CLEARANCE_CUES)


def design_clearance_intent(query: str, uploaded_image_refs) -> bool:
    """Image-gated product-appearance clearance intent.

    True only when the user supplied an uploaded/attached product image AND the
    query carries an appearance-design cue (``has_design_cue``).
    """
    refs = [r for r in (uploaded_image_refs or []) if str(r or "").strip()]
    return bool(refs) and has_design_cue(query)
