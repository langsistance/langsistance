"""Tests for BaitenClient — signature, generic params, streaming base64."""
import asyncio
import base64
import io
import tempfile
import unittest

from sources.baiten_client import (
    BaitenClient,
    _API_METHOD_LAW,
    _API_METHOD_SEARCH,
    _SPOOL_MAX_MEMORY,
    summarize_search_response,
)


class TestTopSignRegression(unittest.TestCase):
    """The signature algorithm is known-vector tested — do not break it."""

    def test_sign_matches_known_vector(self):
        params = {
            "app_key": "test_key",
            "app_num": "CN201080053868",
            "format": "json",
            "law_category": "FSWX",
            "method": "/openService/law",
            "sign_method": "md5",
            "timestamp": "2026-08-01 10:00:00",
            "v": "1.0",
        }
        expected = BaitenClient._top_sign(params, "test_secret")
        # Recompute independently: MD5(secret + sorted k+v concat + secret)
        import hashlib
        concat = "".join(f"{k}{params[k]}" for k in sorted(params) if params[k])
        manual = hashlib.md5(
            f"test_secret{concat}test_secret".encode("UTF-8")).hexdigest().upper()
        self.assertEqual(expected, manual)

    def test_sign_skips_empty_values(self):
        params = {"a": "1", "b": ""}
        sig = BaitenClient._top_sign(params, "s")
        import hashlib
        self.assertEqual(
            sig,
            hashlib.md5(b"s" + b"a1" + b"s").hexdigest().upper(),
        )


class TestBuildTopParams(unittest.TestCase):
    def test_generic_constructor_includes_method_and_sign(self):
        client = BaitenClient("k", "s")
        params = client._build_top_params(
            _API_METHOD_SEARCH,
            {"queryString": "ti:(散热)", "source": "15"},
        )
        self.assertEqual(params["method"], _API_METHOD_SEARCH)
        self.assertEqual(params["queryString"], "ti:(散热)")
        self.assertEqual(params["source"], "15")
        self.assertIn("sign", params)
        self.assertIn("timestamp", params)
        self.assertIn("app_key", params)

    def test_law_uses_same_constructor(self):
        client = BaitenClient("k", "s")
        params = client._build_top_params(
            _API_METHOD_LAW, {"app_num": "CN1", "law_category": "FSWX"})
        self.assertEqual(params["law_category"], "FSWX")


class TestSummarizeSearchResponse(unittest.TestCase):
    """Response summary must distinguish 0 records from unparseable ones."""

    def test_data_wrapper_counts_rows_and_total(self):
        summary = summarize_search_response({
            "code": "200",
            "data": {"total": 37, "fieldValues": [
                {"pn": "CN1"}, {"pn": "CN2"}, {"pn": "CN3"},
            ]},
        })
        self.assertEqual(summary["total"], 37)
        self.assertEqual(summary["rows"], 3)
        self.assertIn("data", summary["keys"])

    def test_top_level_field_values(self):
        summary = summarize_search_response({
            "code": "200",
            "fieldValues": [{"pn": "CN1"}],
        })
        self.assertEqual(summary["total"], None)
        self.assertEqual(summary["rows"], 1)

    def test_empty_and_non_dict_bodies(self):
        self.assertEqual(summarize_search_response({"code": "200"}),
                         {"total": None, "rows": 0, "keys": ["code"]})
        self.assertEqual(summarize_search_response(None),
                         {"total": None, "rows": 0, "keys": []})
        self.assertEqual(summarize_search_response("junk"),
                         {"total": None, "rows": 0, "keys": []})


async def _achunks(payload, size):
    for i in range(0, len(payload), size):
        yield payload[i:i + size]


class TestStreamBase64Field(unittest.TestCase):
    def _decode(self, payload, size=1024, field=b'"fileByte"'):
        out = tempfile.SpooledTemporaryFile(max_size=_SPOOL_MAX_MEMORY)
        found = asyncio.run(BaitenClient._stream_base64_field(
            _achunks(payload, size), field, out))
        out.seek(0)
        data = out.read()
        out.close()
        return found, data

    def test_decodes_across_all_chunk_sizes(self):
        raw = bytes((i % 251) for i in range(50_000))
        b64 = base64.b64encode(raw)
        payload = b'{"code":200,"data":{"fileByte":"' + b64 + b'"}}'
        for size in (1, 3, 5, 7, 1024, 4097):
            found, data = self._decode(payload, size)
            self.assertTrue(found, f"size={size}")
            self.assertEqual(data, raw, f"size={size}")

    def test_returns_false_when_field_absent(self):
        found, data = self._decode(b'{"code":"500","msg":"boom"}')
        self.assertFalse(found)
        self.assertEqual(data, b"")

    def test_returns_false_on_null_value(self):
        found, _ = self._decode(b'{"fileByte":null}')
        self.assertFalse(found)

    def test_peak_memory_stays_bounded(self):
        import tracemalloc
        raw = bytes((i % 251) for i in range(3_000_000))
        b64 = base64.b64encode(raw)
        payload = b'{"fileByte":"' + b64 + b'"}'
        tracemalloc.start()
        found, data = self._decode(payload, 1024)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertTrue(found)
        self.assertEqual(data, raw)
        # 3 MB base64 payload must not balloon memory: stay well under the
        # server's 150 MB streaming budget (memory: server-memory-constraint).
        self.assertLess(peak, 30 * 1024 * 1024)


class TestSpoolAndEncoding(unittest.TestCase):
    def test_encode_params_urlencodes(self):
        encoded = BaitenClient._encode_params({"q": "ti:(a or b)", "k": "v&v"})
        self.assertIn("q=", encoded)
        self.assertIn("ti%3A%28a+or+b%29", encoded)


if __name__ == "__main__":
    unittest.main()
