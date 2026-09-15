# -*- coding: utf-8 -*-
"""需求#29 自产 patent_ids 的可回读性 —— 记录形状与持久化兼容。

背景：面板下发的 CN 号码是**公开号**（`CN…A`），而佰腾法律状态/取件 API
需要的是**申请号**（`app_num`）。旧的持久化只存裸号码字符串，申请号在
提取阶段就被公开号挤掉了，导致系统读不回自己刚产出的号。

本模块的测试锁死两件事：
1. 记录必须携带来源的原生键（native_key），并据此判定可回读性；
2. Redis 读侧必须同时兼容 v1（字符串列表）与 v2（版本化记录）——
   部署顺序安全，滚动重启期间新旧代码共存不会丢数据。
"""
import json
import unittest

from sources.agents.patent_id_records import (
    RECORD_VERSION,
    decode_records,
    encode_records,
    extract_patent_id_records,
    records_from_long_task_rows,
)


class TestExtractRecords(unittest.TestCase):
    def test_baiten_publication_with_app_num_is_retrievable(self):
        item = {"source": "baiten", "patent_id": "CN116570413A",
                "patent_number": "CN116570413A",
                "app_num": "CN202310123456.7", "title": "某方法"}
        recs = extract_patent_id_records([item])
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r["id"], "CN116570413A")
        # 产出侧已是中立值 "cn"（历史值 "baiten" 仍被接受——本文件其余
        # 用例的输入正好覆盖那条兼容路径）。
        self.assertEqual(r["source"], "cn")
        self.assertEqual(r["native_key"], "CN202310123456.7")
        self.assertEqual(r["native_key_kind"], "app_num")
        self.assertTrue(r["retrievable"])

    def test_baiten_publication_without_app_num_not_retrievable(self):
        item = {"source": "baiten", "patent_id": "CN116570413A",
                "patent_number": "CN116570413A"}
        recs = extract_patent_id_records([item])
        self.assertEqual(recs[0]["id"], "CN116570413A")
        self.assertEqual(recs[0]["native_key"], "")
        self.assertFalse(recs[0]["retrievable"])

    def test_uspto_application_number_is_retrievable(self):
        item = {"applicationNumberText": "117941643",
                "applicationMetaData": {"inventionTitle": "A method"}}
        recs = extract_patent_id_records([item])
        r = recs[0]
        self.assertEqual(r["id"], "117941643")
        self.assertEqual(r["source"], "uspto")
        self.assertEqual(r["native_key"], "117941643")
        self.assertEqual(r["native_key_kind"], "applicationNumberText")
        self.assertTrue(r["retrievable"])

    def test_uspto_grant_number_only_is_retrievable(self):
        # 授权号同样可经 applications/search 的引号式文本查询取回
        # （recall probe 已实测 q="11882632" 精确命中）。
        item = {"patentNumber": "11882632",
                "applicationMetaData": {"inventionTitle": "X"}}
        recs = extract_patent_id_records([item])
        r = recs[0]
        self.assertEqual(r["id"], "11882632")
        self.assertEqual(r["source"], "uspto")
        self.assertEqual(r["native_key_kind"], "patentNumber")
        self.assertTrue(r["retrievable"])

    def test_no_key_means_no_key_kind_label(self):
        # 没有键却标着 kind，会诱导下游去解引用一个空键。
        item = {"source": "baiten", "patentNumber": "CN116570413A"}
        recs = extract_patent_id_records([item])
        self.assertEqual(recs[0]["native_key"], "")
        self.assertEqual(recs[0]["native_key_kind"], "")
        self.assertFalse(recs[0]["retrievable"])

    def test_unknown_shape_not_retrievable(self):
        item = {"patent_id": "SOMEOPAQUEID12345"}
        recs = extract_patent_id_records([item])
        self.assertEqual(recs[0]["id"], "SOMEOPAQUEID12345")
        self.assertEqual(recs[0]["native_key"], "")
        self.assertFalse(recs[0]["retrievable"])

    def test_shape_min_width_preserved(self):
        # 与既有 _extract_patent_ids_from_items 的平铺契约一致：
        # 回退分支要求 len>=8。
        recs = extract_patent_id_records([{"patent_id": "SHORT12"}])
        self.assertEqual(recs, [])

    def test_dedupe_preserves_order(self):
        items = [
            {"applicationNumberText": "117941643"},
            {"source": "baiten", "patent_id": "CN116570413A",
             "app_num": "CN202310123456.7"},
            {"applicationNumberText": "117941643"},
        ]
        recs = extract_patent_id_records(items)
        self.assertEqual([r["id"] for r in recs],
                         ["117941643", "CN116570413A"])

    def test_dedupe_merges_native_key_from_later_duplicate(self):
        # 同一个号先以无键形态出现、后以带键形态出现 —— 保留可回读的
        # 那个，不能因为先到者而丢掉原生键。
        items = [
            {"source": "baiten", "patent_id": "CN116570413A"},
            {"source": "baiten", "patent_id": "CN116570413A",
             "app_num": "CN202310123456.7"},
        ]
        recs = extract_patent_id_records(items)
        self.assertEqual(len(recs), 1)
        self.assertTrue(recs[0]["retrievable"])
        self.assertEqual(recs[0]["native_key"], "CN202310123456.7")

    def test_non_dict_items_skipped(self):
        self.assertEqual(extract_patent_id_records([None, "x", 3]), [])

    def test_empty_input(self):
        self.assertEqual(extract_patent_id_records([]), [])
        self.assertEqual(extract_patent_id_records(None), [])


class TestEncodeDecode(unittest.TestCase):
    _REC = {"id": "CN116570413A", "source": "baiten",
            "native_key": "CN202310123456.7", "native_key_kind": "app_num",
            "retrievable": True}

    def test_roundtrip(self):
        blob = encode_records([self._REC])
        self.assertEqual(decode_records(blob), [self._REC])

    def test_encoded_payload_is_versioned(self):
        payload = json.loads(encode_records([self._REC]))
        self.assertEqual(payload["v"], RECORD_VERSION)
        self.assertEqual(payload["ids"], [self._REC])

    def test_decodes_v1_list_of_strings(self):
        # 部署顺序安全：Redis 里可能还是旧格式。
        legacy = json.dumps(["117941643", "CN116570413A"])
        recs = decode_records(legacy)
        self.assertEqual([r["id"] for r in recs],
                         ["117941643", "CN116570413A"])
        self.assertEqual(recs[0]["source"], "")
        self.assertFalse(recs[0]["retrievable"])

    def test_decodes_v2_dict(self):
        blob = json.dumps({"v": 2, "ids": [self._REC]})
        self.assertEqual(decode_records(blob), [self._REC])

    def test_garbage_degrades_to_empty(self):
        for raw in ["", "not json", "{}", "[]", '{"v":2}', None, 123]:
            self.assertEqual(decode_records(raw), [], repr(raw))

    def test_bad_record_entries_skipped(self):
        blob = json.dumps({"v": 2, "ids": [self._REC, None, "x", {}]})
        self.assertEqual(decode_records(blob), [self._REC])

    def test_encode_empty(self):
        self.assertEqual(decode_records(encode_records([])), [])


class TestLongTaskRows(unittest.TestCase):
    """celery 长任务写入方必须与 general_agent 共用同一形状，否则两个
    写入方会互相覆盖成不一致的值。"""

    def test_rows_map_to_records(self):
        rows = [{"专利号": "CN116570413A"}, {"专利号": "CN118453362A"}]
        recs = records_from_long_task_rows(rows, ["专利号"])
        self.assertEqual([r["id"] for r in recs],
                         ["CN116570413A", "CN118453362A"])
        self.assertEqual(recs[0]["source"], "long_task")

    def test_failed_rows_skipped(self):
        rows = [{"专利号": "CN1"}, {"专利号": "CN2", "_failed": True}]
        recs = records_from_long_task_rows(rows, ["专利号"])
        self.assertEqual([r["id"] for r in recs], ["CN1"])

    def test_no_columns_returns_empty(self):
        self.assertEqual(records_from_long_task_rows([{"a": "b"}], []), [])
        self.assertEqual(records_from_long_task_rows([], ["专利号"]), [])

    def test_blank_ids_skipped(self):
        rows = [{"专利号": "  "}, {"专利号": "CN1"}]
        recs = records_from_long_task_rows(rows, ["专利号"])
        self.assertEqual([r["id"] for r in recs], ["CN1"])

    def test_rows_not_retrievable_without_native_key(self):
        # 长任务表只有号码列，没有来源原生键 —— 不能假装可回读。
        recs = records_from_long_task_rows([{"专利号": "CN116570413A"}],
                                           ["专利号"])
        self.assertFalse(recs[0]["retrievable"])


class TestCeleryWriterSharesShape(unittest.TestCase):
    """两个写入方（general_agent / celery_worker）写同一个 Redis 键，
    必须是同一形状 —— 否则后写者会把先写者的值覆盖成对方读不懂的格式。"""

    def _capture(self, rows, columns, user_id="u1"):
        from unittest.mock import MagicMock, patch
        import celery_worker as cw
        captured = {}

        def _set(key, value, ex=None):
            captured["key"] = key
            captured["value"] = value
            captured["ex"] = ex

        fake_r = MagicMock()
        fake_r.set.side_effect = _set
        with patch("sources.knowledge.knowledge.get_redis_connection",
                   return_value=fake_r):
            cw._store_long_task_patent_ids("t1", rows, columns, user_id)
        return captured

    def test_writes_versioned_records(self):
        captured = self._capture([{"专利号": "CN116570413A"}], ["专利号"])
        payload = json.loads(captured["value"])
        self.assertEqual(payload["v"], RECORD_VERSION)
        self.assertEqual(payload["ids"][0]["id"], "CN116570413A")

    def test_preserves_ttl_and_key(self):
        captured = self._capture([{"专利号": "CN1"}], ["专利号"])
        self.assertEqual(captured["key"], "lt:conv:u1:patent_ids")
        self.assertEqual(captured["ex"], 3600)

    def test_written_records_are_readable_by_the_shared_decoder(self):
        # 端到端：celery 写的值，general_agent 的读取路径必须能解。
        captured = self._capture([{"专利号": "CN116570413A"}], ["专利号"])
        recs = decode_records(captured["value"])
        self.assertEqual([r["id"] for r in recs], ["CN116570413A"])


if __name__ == "__main__":
    unittest.main()
