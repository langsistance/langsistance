"""CPC scheme data loader and semantic matcher (plan B, route C).

The Cooperative Patent Classification (CPC) scheme is the official,
data-driven taxonomy patents are classified under.  Matching the user's
question embedding against CPC title embeddings locates the technical
domain ("class"), whose classification language then seeds the search
vocabulary — bridging the recall gap where the query and the patents
share no surface words.

Route context: the USPTO applications/search API does NOT accept CPC
field constraints or bare CPC symbols (probed 2026-08-16), so this
module is used for VOCABULARY expansion only (route C): matched
code+title pairs are handed to the missing-direction prompt.

Design:

- Parser and matcher are pure stdlib + optional numpy — unit-testable
  without network or provider access.
- Vector cache lives on disk (.npy); built once by
  scripts/build_cpc_vectors.py on the server.  Missing cache/file =>
  every function degrades to empty results, never raises.
"""

import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Optional

from sources.logger import Logger

logger = Logger("cpc_semantic.log")

CPC_DATA_DIR = os.getenv("CPC_DATA_DIR", "data/cpc")
CPC_TITLES_JSON = os.path.join(CPC_DATA_DIR, "cpc_titles_main_groups.json")
CPC_VECTORS_NPY = os.path.join(CPC_DATA_DIR, "cpc_title_vectors.npy")

# Main groups only (e.g. H05B45/00) — the coarse domain level.
MAIN_GROUP_RE = re.compile(r"^[A-HY]\d{2}[A-Z]\d{1,4}/00$")


def parse_cpc_zip(zip_path: str, main_groups_only: bool = True) -> list:
    """Extract (code, title) entries from a CPC scheme zip.

    Only the FIRST title-part text of each classification item is kept —
    later title parts carry reference lists, not the title itself.
    *main_groups_only* keeps /00 groups only (the coarse domain level,
    ~14k entries).  Never raises: unreadable archives yield [].
    """
    entries: list = []
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for name in archive.namelist():
                if not name.startswith("cpc-scheme-"):
                    continue
                try:
                    root = ET.fromstring(archive.read(name))
                except ET.ParseError:
                    continue
                for item in root.iter("classification-item"):
                    symbol_el = item.find("classification-symbol")
                    if symbol_el is None or not (symbol_el.text or "").strip():
                        continue
                    code = symbol_el.text.strip()
                    if main_groups_only and not MAIN_GROUP_RE.match(code):
                        continue
                    title_el = item.find("class-title")
                    if title_el is None:
                        continue
                    first_part = title_el.find("title-part")
                    text_el = first_part.find("text") if first_part is not None else None
                    if text_el is None or not (text_el.text or "").strip():
                        continue
                    title = " ".join(text_el.text.split())
                    if title:
                        entries.append({"code": code, "title": title})
    except (OSError, zipfile.BadZipFile):
        logger.warning(f"CPC scheme zip unavailable: {zip_path}")
        return []
    logger.info(f"cpc parse — {len(entries)} entries from {zip_path}")
    return entries


def load_cpc_titles(json_path: str = CPC_TITLES_JSON) -> list:
    """Load parsed CPC titles; [] when the data file is absent."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (OSError, ValueError):
        logger.warning(f"CPC titles file unavailable: {json_path}")
    return []


def load_cpc_vectors(npy_path: str = CPC_VECTORS_NPY) -> Optional[Any]:
    """Load the cached title vectors (.npy); None when absent or numpy
    is not installed."""
    if not os.path.exists(npy_path):
        return None
    try:
        import numpy as np
        return np.load(npy_path)
    except (ImportError, ValueError, OSError):
        logger.warning(f"CPC vector cache unavailable: {npy_path}")
        return None


def match_cpc_codes(query_vector: Any, vectors: Any, entries: list,
                    top_k: int = 8) -> list:
    """Cosine-rank *entries* against *query_vector* using *vectors*.

    Returns up to *top_k* dicts {code, title, score}; [] on any
    degenerate input (missing vectors, length mismatch).
    """
    if query_vector is None or vectors is None or not entries:
        return []
    if len(vectors) != len(entries) or len(vectors) == 0:
        return []
    if len(query_vector) != len(vectors[0]):
        return []
    scores = []
    for vec, entry in zip(vectors, entries):
        dot = sum(float(a) * float(b) for a, b in zip(query_vector, vec))
        norm_a = sum(float(a) * float(a) for a in query_vector) ** 0.5
        norm_b = sum(float(b) * float(b) for b in vec) ** 0.5
        score = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
        scores.append((score, entry))
    scores.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {"code": entry["code"], "title": entry["title"], "score": score}
        for score, entry in scores[:top_k]
    ]


def match_query_to_cpc(query_text: str, top_k: int = 8,
                       extra_terms: Any = "") -> list:
    """Match a user question to CPC main groups end to end.

    Embeds the question AND, when given, *extra_terms* (carrier
    vocabulary from the rewrite stage — a string or a list of
    per-concept term groups, each embedded as its OWN text so no group
    dilutes another) with the configured provider, loads the cached
    titles/vectors, and returns up to *top_k* dicts {code, title,
    score} — per-text matches merged by best score.  Degrades to [] on
    any failure — expansion is an enhancement, never a hard dependency.
    """
    if not query_text or not query_text.strip():
        return []
    entries = load_cpc_titles()
    vectors = load_cpc_vectors()
    if not entries or vectors is None:
        return []
    texts = [query_text]
    if isinstance(extra_terms, (list, tuple)):
        texts.extend(str(t).strip() for t in extra_terms if str(t).strip())
    elif extra_terms and str(extra_terms).strip():
        texts.append(str(extra_terms).strip())
    try:
        from sources.long_task.semantic_rerank import embed_texts
        embedded = embed_texts(texts)
    except Exception:
        return []
    if not embedded or not embedded[0]:
        return []
    best_by_code = {}
    for vec in embedded:
        for m in match_cpc_codes(vec, vectors, entries, top_k=top_k * 2):
            code = m["code"]
            if code not in best_by_code or m["score"] > best_by_code[code]["score"]:
                best_by_code[code] = m
    matches = sorted(best_by_code.values(),
                     key=lambda m: m["score"], reverse=True)[:top_k]
    logger.info(
        f"cpc match — query={query_text[:60]!r} "
        f"top={[m['code'] for m in matches[:5]]}")
    return matches
