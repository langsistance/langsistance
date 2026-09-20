"""Tests for patent family data structures and EPO OPS XML parsing.

Uses the real EPO OPS XML response for US12506212 to validate parsing accuracy.
"""

from __future__ import annotations

import asyncio

import pytest
from sources.long_task.family_member import FamilyMember, PatentFamily
from sources.long_task.patent_family import (
    _parse_family_xml,
    analyzable_jurisdictions,
    EPOError,
)
from sources.long_task.china_examination import (
    _parse_review_decisions,
    adapt_baiten_review_decisions,
)


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


class TestAnalyzableJurisdictions:
    """族分析的范围决策：哪些成员国能支撑一次审查分析。

    背景：`execute_family_analysis` 原先硬取 `get_representative('US')`，
    拿不到就整任务失败——哪怕族里明明有 CN/JP/EP 成员、而这三个国别的审查
    分析器都已存在。用户问「查某中国专利的全球审查历史」时，若该族没有美国
    成员，得到的是「未找到美国同族成员」，与诉求无关。

    这里把「能分析谁」做成**纯函数**，供任务决定继续还是如实报告范围。
    无状态、不碰 I/O，因为它要守的正是"没有美国成员时还能不能干活"。
    """

    @staticmethod
    def _fam(members):
        return PatentFamily(query_pub_number="X", family_id="f", total_count=len(members),
                            members=[FamilyMember(**m) for m in members])

    @staticmethod
    def _member(country, pub="1234567", kind="A1", app="2020123456"):
        return {"country": country, "pub_number": pub, "pub_kind": kind,
                "pub_date": "20200101", "app_number": app, "app_date": "20190101"}

    def test_us_and_cn_members_both_analyzable(self):
        fam = self._fam([self._member("US", kind="B2"),
                         self._member("CN", kind="B", app="201710216936")])
        got = analyzable_jurisdictions(fam)
        assert sorted(got) == ["CN", "US"]
        assert got["US"].country == "US"
        assert got["CN"].country == "CN"

    def test_family_without_us_member_is_still_analyzable(self):
        """纯国内 CN 申请（无美国同族）必须以 CN 为分析对象，而不是失败。"""
        fam = self._fam([self._member("CN", kind="B", app="201710216936")])
        assert sorted(analyzable_jurisdictions(fam)) == ["CN"]

    def test_ep_member_states_fold_into_ep(self):
        """EP 族的成员国（DE/GB/FR…）合并为 EP 一个分析对象。"""
        fam = self._fam([self._member("EP", pub="3000000", kind="A1"),
                         self._member("DE", pub="602012034", kind="T2"),
                         self._member("GB", pub="2500000", kind="A")])
        assert sorted(analyzable_jurisdictions(fam)) == ["EP"]

    def test_unsupported_jurisdictions_are_excluded(self):
        """只有 WO/KR 等无分析器的辖区时，结果为空——调用方据此如实报告。"""
        fam = self._fam([self._member("WO"), self._member("KR")])
        assert analyzable_jurisdictions(fam) == {}

    def test_all_four_offices_supported(self):
        fam = self._fam([self._member("US", kind="B2"),
                         self._member("CN", kind="B"),
                         self._member("EP", pub="3000000"),
                         self._member("JP", kind="B2")])
        assert sorted(analyzable_jurisdictions(fam)) == ["CN", "EP", "JP", "US"]

    def test_granted_member_preferred_within_an_office(self):
        """同一局既有申请公开又有授权件时，取授权件（审查过程更完整）。"""
        fam = self._fam([self._member("US", pub="2020001", kind="A1"),
                         self._member("US", pub="12506212", kind="B2")])
        got = analyzable_jurisdictions(fam)
        assert list(got) == ["US"]
        assert got["US"].pub_kind == "B2"

    def test_empty_family_yields_nothing(self):
        assert analyzable_jurisdictions(self._fam([])) == {}


class TestBaitenDecisionAdapter:
    """佰腾 FSWX 审查决定 → china_examination 的 SIPOP 形态。

    换数据源的动因：CN 审查分析的官方源 SIPOP 不可用，改用佰腾。
    `BaitenClient.query_patent_review` 已经是 "SIPOP-compatible" 签名，
    但**字段名不同**——`_parse_review_decisions` 读的是 SIPOP 的
    camelCase（decisionNumber/decisionDate/appellant…），佰腾给的是
    declareNum/declareDate/reDeclarePerson…。适配层负责这层翻译，
    纯函数、无 I/O，所以可以完整测试。
    """

    def test_core_fields_are_translated(self):
        raw = [{
            "declareNum": "7971", "declareDate": "20240315",
            "inTitle": "一种散热装置", "patentee": "某公司",
            "reDeclarePerson": "请求人甲", "ineffectivePerson": "专利权人乙",
            "mainExamingPerson": "审查员丙", "lawBase": "专利法第22条",
            "mainClassNum": "H05K7/20",
            "fullText": "决定要点：……",
        }]
        out = adapt_baiten_review_decisions(raw)
        assert len(out) == 1
        got = out[0]
        assert got["decisionNumber"] == "7971"
        assert got["decisionDate"] == "20240315"
        assert got["inventionTitle"] == "一种散热装置"
        assert got["assignee"] == "某公司"
        assert got["chiefExaminer"] == "审查员丙"
        assert got["lawReference"] == "专利法第22条"
        assert got["mainClassification"] == "H05K7/20"
        assert got["complainant"] == "请求人甲"
        assert got["defendant"] == "专利权人乙"
        assert got["reasoning"] == "决定要点：……"

    def test_decision_type_inferred_from_full_text(self):
        """佰腾不给决定类型字段——从全文措辞推断，推不出时留空。"""
        cases = [("无效宣告请求审查决定书……", "invalidation"),
                 ("复审请求审查决定书……", "reexamination"),
                 ("驳回决定……", "overrule")]
        for text, expected in cases:
            out = adapt_baiten_review_decisions([{"fullText": text}])
            assert out[0]["decision"] == expected, text

    def test_unknown_decision_type_is_empty_not_guessed(self):
        out = adapt_baiten_review_decisions([{"fullText": "无关文本"}])
        assert out[0]["decision"] == ""

    def test_missing_fields_degrade_to_empty(self):
        out = adapt_baiten_review_decisions([{}])
        assert len(out) == 1
        assert out[0]["decisionNumber"] == ""
        assert out[0]["fullText"] == ""

    def test_non_dict_entries_are_skipped(self):
        out = adapt_baiten_review_decisions(["junk", None, {"declareNum": "1"}])
        assert len(out) == 1

    def test_empty_input(self):
        assert adapt_baiten_review_decisions([]) == []
        assert adapt_baiten_review_decisions(None) == []

    def test_adapted_shape_parses_into_examination_events(self):
        """端到端：适配后的 dict 必须能被 _parse_review_decisions 吃下。"""
        raw = [{"declareNum": "7971", "declareDate": "20240315",
                "inTitle": "T", "reDeclarePerson": "甲",
                "ineffectivePerson": "乙", "fullText": "无效宣告审查决定"}]
        events = _parse_review_decisions(adapt_baiten_review_decisions(raw))
        assert len(events) == 1
        assert events[0].decision_number == "7971"
        assert events[0].decision == "invalidation"
        assert events[0].complainant == "甲"


class TestBaitenExaminationSource:
    """CN 审查分析的数据源：佰腾（审查决定/法律状态）+ 中国专利客户端（著录/全文）。

    单一客户端都凑不齐 `fetch_examination_data` 要的 5 个方法——佰腾有
    review/law_state/legal_timeline，中国专利客户端有 basic_info/full_text。
    组合适配器把两条来源拼成一个接口，使 CN 审查分析不再依赖 SIPOP。
    """

    class _FakeBaiten:
        def __init__(self, decisions=None):
            self.calls = []
            self._decisions = decisions or []

        async def query_patent_review(self, app_num, country="CN"):
            self.calls.append(("review", app_num))
            return self._decisions

        async def query_law_state(self, app_num):
            self.calls.append(("law_state", app_num))
            return {"status": "专利权有效"}

        async def query_legal_state_timeline(self, app_num, country="CN"):
            self.calls.append(("timeline", app_num))
            return [{"date": "20240101", "lawStatus": "授权"}]

    class _FakeCn:
        def __init__(self):
            self.calls = []

        async def query_basic_info(self, app_num):
            self.calls.append(("basic", app_num))
            return {"title": "一种装置"}

        async def query_full_text(self, app_num):
            self.calls.append(("full", app_num))
            return {"claim": ["权利要求1"]}

    @staticmethod
    def _src(baiten=None, cn=None):
        from sources.long_task.china_examination import BaitenExaminationSource
        return BaitenExaminationSource(
            baiten or TestBaitenExaminationSource._FakeBaiten(),
            cn or TestBaitenExaminationSource._FakeCn())

    def test_review_and_law_come_from_baiten(self):
        b = self._FakeBaiten(decisions=[{"declareNum": "1"}])
        src = self._src(baiten=b)
        run = asyncio.run
        assert run(src.query_patent_review("201710216936")) == [{"declareNum": "1"}]
        assert run(src.query_law_state("201710216936")) == {"status": "专利权有效"}
        assert run(src.query_legal_state_timeline("201710216936"))[0]["lawStatus"] == "授权"
        assert [c[0] for c in b.calls] == ["review", "law_state", "timeline"]

    def test_basic_info_and_full_text_come_from_cn_client(self):
        cn = self._FakeCn()
        src = self._src(cn=cn)
        assert asyncio.run(src.query_basic_info("201710216936"))["title"] == "一种装置"
        assert asyncio.run(src.query_full_text("201710216936"))["claim"] == ["权利要求1"]
        assert [c[0] for c in cn.calls] == ["basic", "full"]

    def test_missing_delegate_degrades_to_empty_not_crash(self):
        """任一来源未配置时，该来源的方法返回空——审查分析降级而非中断。"""
        from sources.long_task.china_examination import BaitenExaminationSource
        src = BaitenExaminationSource(None, None)
        run = asyncio.run
        assert run(src.query_patent_review("X")) == []
        assert run(src.query_law_state("X")) == {}
        assert run(src.query_legal_state_timeline("X")) == []
        assert run(src.query_basic_info("X")) == {}
        assert run(src.query_full_text("X")) == {}
