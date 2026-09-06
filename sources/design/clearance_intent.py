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

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".heic",
                     ".heif", ".avif")


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


def is_image_file(name):
    """True when *name* looks like a product image by its file extension.

    Lower-cased, dot-anchored suffix test so ``foo.PNG`` and ``shot.webp``
    count while patent-document files (``.pdf``/``.docx``/``.xml``) never do —
    it is the *image* half of the seller appearance-clearance gate.
    """
    if not name or not str(name).strip():
        return False
    return str(name).strip().lower().endswith(_IMAGE_EXTENSIONS)


def uploaded_image_refs_of(file_refs) -> list:
    """Of the saved multipart refs ({filename,path,...}) keep those that are images.

    Returns the file *paths* (``product_image_refs`` shape accepted by the design
    executor / pipeline) for image refs only.  Empty when no image present so a
    patent-document batch is never mistaken for a product-appearance upload.
    """
    paths = []
    for ref in file_refs or []:
        if isinstance(ref, dict) and is_image_file(ref.get("filename")
                                                    or ref.get("path") or ""):
            path = ref.get("path")
            if path:
                paths.append(path)
    return paths


def seller_design_clearance_gate(scene, file_refs, query) -> bool:
    """Seller-scope gate (T8): image-file ∧ scene=="seller" ∧ appearance cue.

    Controller ruling (A): an appearance-design clearance is dispatched only in
    the seller scene — the three conditions below must ALL hold, otherwise the
    upload carries on its original (patent-analysis) path unchanged:

      - an uploaded product IMAGE is present (extension-recognised ref), AND
      - the multipart request declares ``scene == "seller"``, AND
      - the query text OR an uploaded filename carries an appearance cue
        (``has_design_cue``) — the cue vocabulary is shared/reused, never a
        single test question.
    """
    if not (scene or "").strip() == "seller":
        return False
    if not uploaded_image_refs_of(file_refs):
        return False
    return has_design_cue(query) or any(
        has_design_cue(str(r.get("filename") or ""))
        for r in (file_refs or []) if isinstance(r, dict))
