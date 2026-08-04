"""Patent family data structures parsed from EPO OPS family API responses.

A patent family is a group of patent documents in different jurisdictions
that share the same priority claim(s) — i.e. the same invention filed in
multiple countries.

Typical flow:
    EPO OPS XML response → PatentFamily.from_ops_xml() → FamilyMember list
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Kind-code → status mapping ──────────────────────────────────────────────────

# These mappings encode whether a given DOCDB kind code represents a granted
# (B-level) patent, a pre-grant publication (A-level), or something else.
# They are deliberately conservative — only codes we have verified across
# US/CN/JP/EP/WO are listed.  Unrecognised codes default to 'unknown'.
_GRANTED_KINDS: dict[str, set[str]] = {
    "US": {"B1", "B2", "C1", "C2", "C3"},
    "CN": {"B", "C"},
    "JP": {"B1", "B2", "B", "C"},
    "EP": {"B1", "B2"},
    "WO": set(),  # PCT publications are never grants
}
_APPLICATION_KINDS: dict[str, set[str]] = {
    "US": {"A1", "A2", "A9"},
    "CN": {"A"},
    "JP": {"A", "A1", "A2"},
    "EP": {"A1", "A2"},
    "WO": {"A1", "A2", "A3"},
}


# ── Data classes ─────────────────────────────────────────────────────────────────


@dataclass
class FamilyMember:
    """A single patent-family member from an EPO OPS family response.

    Each family member is one publication (application or grant) in one
    jurisdiction.  Multiple members may share the same application-number,
    representing different publication stages (e.g. A1 → B2).
    """

    country: str          # two-letter code: US, CN, JP, EP, WO, ...
    pub_number: str       # publication / grant number (e.g. "12506212")
    pub_kind: str         # DOCDB kind code (e.g. "B2", "A1", "A")
    pub_date: str         # YYYYMMDD (e.g. "20251223")
    app_number: str       # DOCDB application number (e.g. "202017638216")
    app_date: str         # YYYYMMDD filing date
    title: str = ""       # English invention title (from exchange-document)

    @property
    def is_granted(self) -> bool:
        """Whether this publication is a granted patent (B/C-level)."""
        allowed = _GRANTED_KINDS.get(self.country, set())
        return self.pub_kind in allowed

    @property
    def is_application(self) -> bool:
        """Whether this publication is a pre-grant application publication."""
        allowed = _APPLICATION_KINDS.get(self.country, set())
        return self.pub_kind in allowed

    @property
    def normalized_app_number(self) -> str:
        """Application number in a form usable for USPTO / CNIPA API queries.

        For USPTO, strips leading country-year prefix to get the serial number.
        For other countries, returns the raw DOCDB app_number.
        """
        if self.country == "US":
            # DOCDB US app numbers look like "202017638216".
            # The USPTO API expects the 8-digit serial "17638216".
            # The pattern is: 4-digit year + 8-digit serial.
            if len(self.app_number) == 12 and self.app_number.isdigit():
                return self.app_number[4:]
            if len(self.app_number) == 8 and self.app_number.isdigit():
                return self.app_number
        return self.app_number

    @property
    def family_key(self) -> str:
        """Stable key for deduplication: ``country + app_number``.

        Two family members that share a country + application number are the
        same patent at different publication stages (e.g. application + grant).
        """
        return f"{self.country}:{self.app_number}"


@dataclass
class PatentFamily:
    """A complete patent family returned by the EPO OPS family API."""

    query_pub_number: str               # the publication number used to query
    family_id: str                      # EPO internal family id
    total_count: int                    # raw count from XML (includes duplicates)
    members: list[FamilyMember] = field(default_factory=list)

    # ── Computed helpers ──────────────────────────────────────────────────────

    @property
    def jurisdictions(self) -> list[str]:
        """Deduplicated, sorted list of two-letter country codes in this family."""
        seen: dict[str, bool] = {}
        for m in self.members:
            if m.country not in seen:
                seen[m.country] = True
        return sorted(seen.keys())

    @property
    def deduplicated_members(self) -> list[FamilyMember]:
        """Members deduplicated by ``family_key``, preferring grants over applications.

        When the same application appears as both an A1 (application) and B2
        (grant), keep the grant.  Otherwise keep the first occurrence.
        """
        by_key: dict[str, FamilyMember] = {}
        for m in self.members:
            key = m.family_key
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = m
            elif m.is_granted and not existing.is_granted:
                # Prefer the granted version
                by_key[key] = m
            elif m.is_application and existing.is_application and m.pub_date > existing.pub_date:
                # Both are applications — keep the newest
                by_key[key] = m
        return list(by_key.values())

    def for_jurisdiction(self, country: str) -> list[FamilyMember]:
        """Return all members for *country*, newest first."""
        matches = [m for m in self.deduplicated_members if m.country == country]
        matches.sort(key=lambda m: m.pub_date, reverse=True)
        return matches

    def get_representative(self, country: str) -> FamilyMember | None:
        """Best single member for *country*: grant if available, else newest app."""
        members = self.for_jurisdiction(country)
        if not members:
            return None
        # Prefer granted
        for m in members:
            if m.is_granted:
                return m
        return members[0]  # newest application
