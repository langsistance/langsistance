"""Tests for patent family data structures and EPO OPS XML parsing.

Uses the real EPO OPS XML response for US12506212 to validate parsing accuracy.
"""

from __future__ import annotations

import pytest
from sources.long_task.family_member import FamilyMember, PatentFamily
from sources.long_task.patent_family import _parse_family_xml, EPOError


# ── Real EPO OPS XML response for US12506212 (trimmed to essential elements) ───

EPO_XML_US12506212 = """<?xml version="1.0" encoding="UTF-8"?>
<ops:world-patent-data xmlns="http://www.epo.org/exchange"
                       xmlns:ops="http://ops.epo.org"
                       xmlns:xlink="http://www.w3.org/1999/xlink">
  <ops:patent-family legal="false" total-result-count="7">
    <ops:publication-reference>
      <document-id document-id-type="docdb">
        <country>US</country>
        <doc-number>12506212</doc-number>
        <kind>%%</kind>
      </document-id>
    </ops:publication-reference>

    <!-- US application publication -->
    <ops:family-member family-id="74847988">
      <publication-reference>
        <document-id document-id-type="docdb">
          <country>US</country>
          <doc-number>2022294065</doc-number>
          <kind>A1</kind>
          <date>20220915</date>
        </document-id>
      </publication-reference>
      <application-reference doc-id="579503025" is-representative="YES">
        <document-id document-id-type="docdb">
          <country>US</country>
          <doc-number>202017638216</doc-number>
          <kind>A</kind>
          <date>20200902</date>
        </document-id>
      </application-reference>
      <exchange-document system="ops.epo.org" family-id="74847988" country="US"
                         doc-number="2022294065" kind="A1">
        <bibliographic-data>
          <invention-title lang="en">Secondary battery accommodating structure and humanoid robot</invention-title>
        </bibliographic-data>
      </exchange-document>
    </ops:family-member>

    <!-- US granted patent -->
    <ops:family-member family-id="74847988">
      <publication-reference>
        <document-id document-id-type="docdb">
          <country>US</country>
          <doc-number>12506212</doc-number>
          <kind>B2</kind>
          <date>20251223</date>
        </document-id>
      </publication-reference>
      <application-reference doc-id="579503025" is-representative="YES">
        <document-id document-id-type="docdb">
          <country>US</country>
          <doc-number>202017638216</doc-number>
          <kind>A</kind>
          <date>20200902</date>
        </document-id>
      </application-reference>
      <exchange-document system="ops.epo.org" family-id="74847988" country="US"
                         doc-number="12506212" kind="B2">
        <bibliographic-data>
          <invention-title lang="en">SECONDARY BATTERY ACCOMMODATING STRUCTURE AND HUMANOID ROBOT</invention-title>
        </bibliographic-data>
      </exchange-document>
    </ops:family-member>

    <!-- CN application publication -->
    <ops:family-member family-id="74847988">
      <publication-reference>
        <document-id document-id-type="docdb">
          <country>CN</country>
          <doc-number>114340854</doc-number>
          <kind>A</kind>
          <date>20220412</date>
        </document-id>
      </publication-reference>
      <application-reference doc-id="570474367">
        <document-id document-id-type="docdb">
          <country>CN</country>
          <doc-number>202080061975</doc-number>
          <kind>A</kind>
          <date>20200902</date>
        </document-id>
      </application-reference>
      <exchange-document system="ops.epo.org" family-id="74847988" country="CN"
                         doc-number="114340854" kind="A">
        <bibliographic-data>
          <invention-title lang="en">Storage battery accommodating structure and humanoid robot</invention-title>
        </bibliographic-data>
      </exchange-document>
    </ops:family-member>

    <!-- CN granted patent -->
    <ops:family-member family-id="74847988">
      <publication-reference>
        <document-id document-id-type="docdb">
          <country>CN</country>
          <doc-number>114340854</doc-number>
          <kind>B</kind>
          <date>20230721</date>
        </document-id>
      </publication-reference>
      <application-reference doc-id="570474367">
        <document-id document-id-type="docdb">
          <country>CN</country>
          <doc-number>202080061975</doc-number>
          <kind>A</kind>
          <date>20200902</date>
        </document-id>
      </application-reference>
    </ops:family-member>

    <!-- JP application publication -->
    <ops:family-member family-id="74847988">
      <publication-reference>
        <document-id document-id-type="docdb">
          <country>JP</country>
          <doc-number>2021037573</doc-number>
          <kind>A</kind>
          <date>20210311</date>
        </document-id>
      </publication-reference>
      <application-reference doc-id="547012526">
        <document-id document-id-type="docdb">
          <country>JP</country>
          <doc-number>2019159764</doc-number>
          <kind>A</kind>
          <date>20190902</date>
        </document-id>
      </application-reference>
      <exchange-document system="ops.epo.org" family-id="74847988" country="JP"
                         doc-number="2021037573" kind="A">
        <bibliographic-data>
          <invention-title lang="en">STORAGE BATTERY HOUSING STRUCTURE AND HUMANOID ROBOT</invention-title>
        </bibliographic-data>
      </exchange-document>
    </ops:family-member>

    <!-- JP granted patent -->
    <ops:family-member family-id="74847988">
      <publication-reference>
        <document-id document-id-type="docdb">
          <country>JP</country>
          <doc-number>7274385</doc-number>
          <kind>B2</kind>
          <date>20230516</date>
        </document-id>
      </publication-reference>
      <application-reference doc-id="547012526">
        <document-id document-id-type="docdb">
          <country>JP</country>
          <doc-number>2019159764</doc-number>
          <kind>A</kind>
          <date>20190902</date>
        </document-id>
      </application-reference>
    </ops:family-member>

  </ops:patent-family>
</ops:world-patent-data>"""


# ── Tests ───────────────────────────────────────────────────────────────────────


class TestFamilyMember:
    """Tests for FamilyMember dataclass properties."""

    def test_is_granted_us_b2(self):
        m = FamilyMember(country="US", pub_number="12506212", pub_kind="B2",
                          pub_date="20251223", app_number="202017638216", app_date="20200902")
        assert m.is_granted is True
        assert m.is_application is False

    def test_is_granted_us_a1(self):
        m = FamilyMember(country="US", pub_number="2022294065", pub_kind="A1",
                          pub_date="20220915", app_number="202017638216", app_date="20200902")
        assert m.is_granted is False
        assert m.is_application is True

    def test_is_granted_cn_b(self):
        m = FamilyMember(country="CN", pub_number="114340854", pub_kind="B",
                          pub_date="20230721", app_number="202080061975", app_date="20200902")
        assert m.is_granted is True

    def test_is_granted_jp_b2(self):
        m = FamilyMember(country="JP", pub_number="7274385", pub_kind="B2",
                          pub_date="20230516", app_number="2019159764", app_date="20190902")
        assert m.is_granted is True

    def test_is_granted_jp_a(self):
        m = FamilyMember(country="JP", pub_number="2021037573", pub_kind="A",
                          pub_date="20210311", app_number="2019159764", app_date="20190902")
        assert m.is_granted is False
        assert m.is_application is True

    def test_is_granted_wo_never(self):
        m = FamilyMember(country="WO", pub_number="2022036365", pub_kind="A1",
                          pub_date="20220224", app_number="PCT12345", app_date="20200801")
        assert m.is_granted is False  # PCT publications are never grants

    def test_is_granted_unknown_country(self):
        m = FamilyMember(country="XX", pub_number="12345", pub_kind="B1",
                          pub_date="20200101", app_number="67890", app_date="20190101")
        assert m.is_granted is False  # Unrecognised country → default to not granted

    def test_normalized_app_number_us_12_digit(self):
        m = FamilyMember(country="US", pub_number="12506212", pub_kind="B2",
                          pub_date="20251223", app_number="202017638216", app_date="20200902")
        assert m.normalized_app_number == "17638216"

    def test_normalized_app_number_us_8_digit(self):
        m = FamilyMember(country="US", pub_number="17429113", pub_kind="A1",
                          pub_date="20230101", app_number="17429113", app_date="20220101")
        assert m.normalized_app_number == "17429113"

    def test_normalized_app_number_non_us(self):
        m = FamilyMember(country="CN", pub_number="114340854", pub_kind="B",
                          pub_date="20230721", app_number="202080061975", app_date="20200902")
        assert m.normalized_app_number == "202080061975"

    def test_family_key(self):
        m1 = FamilyMember(country="US", pub_number="2022294065", pub_kind="A1",
                           pub_date="20220915", app_number="202017638216", app_date="20200902")
        m2 = FamilyMember(country="US", pub_number="12506212", pub_kind="B2",
                           pub_date="20251223", app_number="202017638216", app_date="20200902")
        assert m1.family_key == m2.family_key
        assert m1.family_key == "US:202017638216"


class TestPatentFamily:
    """Tests for PatentFamily computed properties."""

    @pytest.fixture
    def us12506212_family(self):
        return _parse_family_xml(EPO_XML_US12506212, "US12506212")

    def test_parse_basic_info(self, us12506212_family):
        assert us12506212_family.query_pub_number == "US12506212"
        assert us12506212_family.family_id == "74847988"
        assert us12506212_family.total_count == 7

    def test_parse_has_correct_jurisdictions(self, us12506212_family):
        assert us12506212_family.jurisdictions == ["CN", "JP", "US"]

    def test_parse_all_members(self, us12506212_family):
        assert len(us12506212_family.members) == 6  # 6 family-member elements in test XML

    def test_deduplicated_members(self, us12506212_family):
        dedup = us12506212_family.deduplicated_members
        # 3 jurisdictions × 1 representative each = 3 members after dedup
        assert len(dedup) == 3

    def test_deduplicate_prefers_granted(self, us12506212_family):
        dedup = us12506212_family.deduplicated_members
        us = [m for m in dedup if m.country == "US"]
        assert len(us) == 1
        assert us[0].pub_kind == "B2"  # grant, not A1

    def test_for_jurisdiction_us(self, us12506212_family):
        us_members = us12506212_family.for_jurisdiction("US")
        assert len(us_members) == 1
        assert us_members[0].pub_kind == "B2"
        assert us_members[0].pub_number == "12506212"

    def test_for_jurisdiction_cn(self, us12506212_family):
        cn_members = us12506212_family.for_jurisdiction("CN")
        assert len(cn_members) == 1
        assert cn_members[0].pub_kind == "B"

    def test_for_jurisdiction_jp(self, us12506212_family):
        jp_members = us12506212_family.for_jurisdiction("JP")
        assert len(jp_members) == 1
        assert jp_members[0].pub_kind == "B2"

    def test_get_representative_us(self, us12506212_family):
        rep = us12506212_family.get_representative("US")
        assert rep is not None
        assert rep.is_granted is True
        assert rep.pub_number == "12506212"
        assert rep.normalized_app_number == "17638216"

    def test_get_representative_cn(self, us12506212_family):
        rep = us12506212_family.get_representative("CN")
        assert rep is not None
        assert rep.is_granted is True
        assert rep.pub_number == "114340854"

    def test_get_representative_jp(self, us12506212_family):
        rep = us12506212_family.get_representative("JP")
        assert rep is not None
        assert rep.is_granted is True
        assert rep.pub_number == "7274385"

    def test_get_representative_nonexistent_country(self, us12506212_family):
        assert us12506212_family.get_representative("KR") is None

    def test_for_jurisdiction_nonexistent(self, us12506212_family):
        assert us12506212_family.for_jurisdiction("KR") == []

    def test_title_extraction(self, us12506212_family):
        us = us12506212_family.get_representative("US")
        assert "SECONDARY BATTERY" in us.title.upper()


class TestXMLParserEdgeCases:
    """Edge case tests for XML parsing."""

    def test_empty_family(self):
        xml = """<?xml version="1.0"?>
        <ops:world-patent-data xmlns="http://www.epo.org/exchange"
                               xmlns:ops="http://ops.epo.org">
          <ops:patent-family legal="false" total-result-count="0">
            <ops:publication-reference>
              <document-id document-id-type="docdb">
                <country>US</country><doc-number>99999999</doc-number><kind>%%</kind>
              </document-id>
            </ops:publication-reference>
          </ops:patent-family>
        </ops:world-patent-data>"""
        family = _parse_family_xml(xml, "US99999999")
        assert family.total_count == 0
        assert len(family.members) == 0
        assert family.jurisdictions == []

    def test_missing_patent_family_raises(self):
        xml = """<?xml version="1.0"?>
        <ops:world-patent-data xmlns="http://www.epo.org/exchange"
                               xmlns:ops="http://ops.epo.org">
        </ops:world-patent-data>"""
        with pytest.raises(EPOError, match="patent-family"):
            _parse_family_xml(xml, "US12345")

    def test_member_without_publication_reference_is_skipped(self):
        xml = """<?xml version="1.0"?>
        <ops:world-patent-data xmlns="http://www.epo.org/exchange"
                               xmlns:ops="http://ops.epo.org">
          <ops:patent-family legal="false" total-result-count="2">
            <ops:publication-reference>
              <document-id document-id-type="docdb">
                <country>US</country><doc-number>12345</doc-number><kind>%%</kind>
              </document-id>
            </ops:publication-reference>
            <ops:family-member family-id="1">
              <publication-reference>
                <document-id document-id-type="docdb">
                  <country>US</country><doc-number>12345</doc-number><kind>A1</kind><date>20200101</date>
                </document-id>
              </publication-reference>
              <application-reference>
                <document-id document-id-type="docdb">
                  <country>US</country><doc-number>11111111</doc-number><kind>A</kind>
                </document-id>
              </application-reference>
            </ops:family-member>
            <!-- Member with no publication-reference: should be skipped -->
            <ops:family-member family-id="1">
            </ops:family-member>
          </ops:patent-family>
        </ops:world-patent-data>"""
        family = _parse_family_xml(xml, "US12345")
        assert len(family.members) == 1  # only the valid one

    def test_member_without_docdb_document_id_is_skipped(self):
        xml = """<?xml version="1.0"?>
        <ops:world-patent-data xmlns="http://www.epo.org/exchange"
                               xmlns:ops="http://ops.epo.org">
          <ops:patent-family legal="false" total-result-count="2">
            <ops:publication-reference>
              <document-id document-id-type="docdb">
                <country>US</country><doc-number>12345</doc-number><kind>%%</kind>
              </document-id>
            </ops:publication-reference>
            <ops:family-member family-id="1">
              <publication-reference>
                <document-id document-id-type="docdb">
                  <country>US</country><doc-number>12345</doc-number><kind>A1</kind><date>20200101</date>
                </document-id>
              </publication-reference>
              <application-reference>
                <document-id document-id-type="docdb">
                  <country>US</country><doc-number>11111111</doc-number><kind>A</kind>
                </document-id>
              </application-reference>
            </ops:family-member>
            <!-- Member with only epodoc doc-id (no docdb): skipped -->
            <ops:family-member family-id="1">
              <publication-reference>
                <document-id document-id-type="epodoc">
                  <doc-number>US99999999</doc-number><date>20200101</date>
                </document-id>
              </publication-reference>
            </ops:family-member>
          </ops:patent-family>
        </ops:world-patent-data>"""
        family = _parse_family_xml(xml, "US12345")
        assert len(family.members) == 1  # epodoc-only member skipped

    def test_family_id_from_first_member(self):
        """family_id should be taken from the first family-member element."""
        xml = """<?xml version="1.0"?>
        <ops:world-patent-data xmlns="http://www.epo.org/exchange"
                               xmlns:ops="http://ops.epo.org">
          <ops:patent-family legal="false" total-result-count="1">
            <ops:publication-reference>
              <document-id document-id-type="docdb">
                <country>US</country><doc-number>12345</doc-number><kind>%%</kind>
              </document-id>
            </ops:publication-reference>
            <ops:family-member family-id="99999999">
              <publication-reference>
                <document-id document-id-type="docdb">
                  <country>US</country><doc-number>12345</doc-number><kind>A1</kind><date>20200101</date>
                </document-id>
              </publication-reference>
              <application-reference>
                <document-id document-id-type="docdb">
                  <country>US</country><doc-number>11111111</doc-number><kind>A</kind>
                </document-id>
              </application-reference>
            </ops:family-member>
          </ops:patent-family>
        </ops:world-patent-data>"""
        family = _parse_family_xml(xml, "US12345")
        assert family.family_id == "99999999"
