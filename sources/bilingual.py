"""Bilingual text (zh:...|en:...) parsing utilities.

Used by API endpoints to return single-language content from bilingual
database fields based on the user's language preference.
"""

import re
from typing import Optional

# Regex to split bilingual text ONLY on | followed by zh: or en:
# This avoids false splits when | appears inside the content itself.
_BILINGUAL_SPLIT_RE = re.compile(r'\|(?=(?:zh|en):)')


def pick_lang(text: Optional[str], lang: str) -> str:
    """Parse bilingual ``zh:Chinese|en:English`` text and return the *lang* part.

    Returns the original text unchanged when no pipe separator is found
    (backward compatible with single-language data).

    Args:
        text: Bilingual text in ``zh:...|en:...`` format, or single-language text.
        lang: Language code to extract (``"zh"`` or ``"en"``).

    Returns:
        The text for the requested language, with the language prefix stripped.
    """
    if not text:
        return ''

    # Normalize lang to base form (strip country code if present)
    lang = lang.split('-')[0].lower() if lang else 'zh'
    if lang not in ('zh', 'en'):
        lang = 'zh'  # default fallback

    # ── Step 1: split ONLY on | that is followed by zh: or en: ──────────
    # This avoids false splits when | appears inside the content itself.
    parts = _BILINGUAL_SPLIT_RE.split(text)

    prefix = f'{lang}:'
    other_prefix = 'zh:' if lang == 'en' else 'en:'

    # ── Step 2: look for the segment starting with the requested lang ───
    for part in parts:
        part = part.strip()
        if part.startswith(prefix):
            result = part[len(prefix):]

            # ── Step 3: strip any embedded other-language marker ─────────
            # If the data is corrupted and the other language's prefix
            # appears without a | delimiter, truncate at that point.
            idx = result.find(other_prefix)
            if idx != -1:
                result = result[:idx].rstrip()

            # Also handle the case where the SAME prefix appears again
            # (double-applied) — strip everything after any re-appearance.
            idx2 = result.find(f'|{prefix}')
            if idx2 != -1:
                result = result[:idx2].rstrip()
            idx3 = result.find(prefix)
            if idx3 != -1:
                # Only strip if it's a plausible embedded marker (not part
                # of normal text — zh:/en: are unlikely in normal content)
                result = result[:idx3].rstrip()

            return result

    # ── Fallback: single-language data (no | delimiter found) ─────────
    # Strip any leading language prefix if present, and also try to
    # truncate at embedded other-language markers (corrupted data).
    for pfx in ('zh:', 'en:'):
        if text.startswith(pfx):
            result = text[len(pfx):]
            # If the other language marker appears embedded (corrupted data
            # where the | separator is missing), truncate at that point.
            other = 'en:' if pfx == 'zh:' else 'zh:'
            idx = result.find(other)
            if idx != -1:
                result = result[:idx].rstrip()
            return result
    return text


def localize_knowledge_fields(item_dict: dict, lang: str) -> dict:
    """Apply bilingual parsing to knowledge question + description fields in place.

    Modifies the dict in place and returns it for convenience.
    """
    item_dict['question'] = pick_lang(item_dict.get('question', ''), lang)
    item_dict['description'] = pick_lang(item_dict.get('description', ''), lang)
    return item_dict


def localize_scene_fields(scene_dict: dict, lang: str) -> dict:
    """Apply bilingual parsing to scene name + description fields in place.

    Modifies the dict in place and returns it for convenience.
    """
    scene_dict['name'] = pick_lang(scene_dict.get('name', ''), lang)
    scene_dict['description'] = pick_lang(scene_dict.get('description', ''), lang)
    return scene_dict
