"""Tests for BaitenClient — signature, generic params, streaming base64."""
import asyncio
import base64
import io
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs

from sources.baiten_client import (
    BaitenClient,
    _API_METHOD_LAW,
    _API_METHOD_SEARCH,
    _REQUEST_HEADERS,
    _SPOOL_MAX_MEMORY,
    _compute_client_sign,
    _error_preview,
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


class TestRequestJson(unittest.TestCase):
    """Regression: _REQUEST_HEADERS is a module constant — referencing it as
    self._REQUEST_HEADERS raised AttributeError on every request, so the
    client never reached the gateway (production 2026-08-26)."""

    async def _run(self):
        class _FakeResponse:
            status_code = 200
            text = '{"code": "200", "data": {"fieldValues": []}}'

            def json(self):
                return {"code": "200", "data": {"fieldValues": []}}

        calls = []

        class _FakeClient:
            async def post(self, url, content, headers):
                calls.append((url, headers))
                return _FakeResponse()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        with patch("sources.baiten_client.httpx.AsyncClient",
                   return_value=_FakeClient()):
            client = BaitenClient("k", "s", "http://gw")
            return await client._request_json(
                _API_METHOD_SEARCH, {"queryString": "ti:(散热)"}), calls

    def test_post_reaches_gateway_with_module_headers(self):
        body, calls = asyncio.run(self._run())
        self.assertEqual(body["code"], "200")
        self.assertEqual(len(calls), 1)
        url, headers = calls[0]
        self.assertIn("/extService/search", url)
        self.assertEqual(headers, _REQUEST_HEADERS)


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

    def test_live_gateway_shape_grouped_hits(self):
        # Error envelope observed on the live gateway (2026-08-26):
        # {qTime, total_hits, grouped_hits, code, msg}.
        summary = summarize_search_response({
            "qTime": 0, "total_hits": 152,
            "grouped_hits": [{"id": 1}, {"id": 2}],
            "code": "200", "msg": "ok",
        })
        self.assertEqual(summary["total"], 152)
        self.assertEqual(summary["rows"], 2)

    def test_documented_success_shape_documents(self):
        # 2023 API docs: search returns {qTime, totalHits, documents}.
        summary = summarize_search_response({
            "qTime": 1, "totalHits": 3, "documents": [{"a": 1}, {"a": 2}],
        })
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["rows"], 2)


class TestComputeClientSign(unittest.TestCase):
    """Known-vector regression for the SDK DefaultCubeClient.doPost
    algorithm (bytecode-reverse-engineered 2026-08-26):
    MD5(dateStr + str(len(query.strip())) + appSecret), dateStr =
    "yyyy-MM-dd HH:mm:ss" GMT+8, uppercase hex."""

    def test_known_vector(self):
        import hashlib
        from datetime import datetime, timezone, timedelta
        now = datetime(2026, 8, 26, 5, 30, 0,
                       tzinfo=timezone(timedelta(hours=8)))
        # len("ti:(散热)") = 7（t i : ( 散 热 )）
        sign = _compute_client_sign("ti:(散热)", "secret123", now=now)
        expected = hashlib.md5(
            ("2026-08-26 05:30:00" + "7" + "secret123").encode("UTF-8"),
        ).hexdigest().upper()
        self.assertEqual(sign, expected)

    def test_trims_whitespace_in_length(self):
        from datetime import datetime, timezone, timedelta
        now = datetime(2026, 8, 26, 5, 30, 0,
                       tzinfo=timezone(timedelta(hours=8)))
        # Java trim(): "  ti:(散热)  " → len 7
        sign = _compute_client_sign("  ti:(散热)  ", "s", now=now)
        import hashlib
        expected = hashlib.md5(
            ("2026-08-26 05:30:00" + "7" + "s").encode("UTF-8"),
        ).hexdigest().upper()
        self.assertEqual(sign, expected)


class TestErrorPreview(unittest.TestCase):
    """HTML error pages from the Spring gateway must yield their Message."""

    def test_extracts_spring_required_param_message(self):
        html = ('<!doctype html><html lang="en"><head><title>HTTP Status 400 '
                '– Bad Request</title><style>body {font-family:Tahoma}</style>'
                '</head><body><h1>HTTP Status 400 – Bad Request</h1><p><b>'
                'Message</b> Required String parameter &#39;level&#39; is not '
                'present</p></body></html>')
        preview = _error_preview(html)
        self.assertEqual(
            preview, "Required String parameter 'level' is not present")

    def test_extracts_title_when_no_message(self):
        preview = _error_preview(
            "<html><head><title>HTTP Status 404 – Not Found</title></head></html>")
        self.assertEqual(preview, "HTTP Status 404 – Not Found")

    def test_plain_text_truncated(self):
        preview = _error_preview("x" * 500)
        self.assertEqual(len(preview), 300)

    def test_empty_body(self):
        self.assertEqual(_error_preview(""), "(empty body)")


class TestLiveWireParams(unittest.TestCase):
    """Wire contracts verified against the live gateway 2026-08-26: search
    wants query/level/page_index/page_size, claims app_num/pat_type,
    download pub_num/pub_date — the SDK-style names get a Spring 400."""

    def test_search_wire_param_names(self):
        # SDK-native surface: /extService/search with query/page_index/
        # page_size/fields/client_sign — no level (the /openService alias
        # requires it and gates on DATA_PAT_BASE_* product permission).
        fake = _FakeHttpClient(status=200, body={"code": "200"})
        with patch("sources.baiten_client.httpx.AsyncClient",
                   return_value=fake):
            client = BaitenClient("k", "s", "http://gw")
            asyncio.run(client.search("ti:(散热)", page=2, page_size=30))
        call = fake.calls[0]
        self.assertIn("/extService/search", call["url"])
        params = parse_qs(call["content"])
        self.assertEqual(params["query"], ["ti:(散热)"])
        self.assertEqual(params["page_index"], ["2"])
        self.assertEqual(params["page_size"], ["30"])
        self.assertEqual(params["source"], ["15"])
        self.assertEqual(params["fields"], ["ti,pa,an,pn,pd,ad,ab"])
        self.assertNotIn("level", params)
        self.assertNotIn("apiLevel", params)
        self.assertIn("client_sign", params)
        self.assertEqual(len(params["client_sign"][0]), 32)  # MD5 hex
        self.assertTrue(params["client_sign"][0].isupper())

    def test_search_fields_override(self):
        fake = _FakeHttpClient(status=200, body={"code": "200"})
        with patch("sources.baiten_client.httpx.AsyncClient",
                   return_value=fake):
            client = BaitenClient("k", "s", "http://gw")
            asyncio.run(client.search("ti:(散热)", fields="ti,pa"))
        params = parse_qs(fake.calls[0]["content"])
        self.assertEqual(params["fields"], ["ti,pa"])

    def test_claims_wire_param_names(self):
        fake = _FakeHttpClient(status=200, body={"code": "200"})
        with patch("sources.baiten_client.httpx.AsyncClient",
                   return_value=fake):
            client = BaitenClient("k", "s", "http://gw")
            asyncio.run(client.get_claims("CN1", "AUTH"))
        call = fake.calls[0]
        self.assertIn("/openService/claims", call["url"])
        params = parse_qs(call["content"])
        self.assertEqual(params["app_num"], ["CN1"])
        self.assertEqual(params["pat_type"], ["AUTH"])
        self.assertNotIn("patType", params)
        self.assertNotIn("appNum", params)

    def test_get_doc_wire_params_and_path(self):
        # SDK-native path: /extService/get with doc_id + client_sign.
        fake = _FakeHttpClient(status=200, body={"code": "200"})
        with patch("sources.baiten_client.httpx.AsyncClient",
                   return_value=fake):
            client = BaitenClient("k", "s", "http://gw")
            asyncio.run(client.get_doc("CN118000001A", client_sign="abc"))
        call = fake.calls[0]
        self.assertIn("/extService/get", call["url"])
        params = parse_qs(call["content"])
        self.assertEqual(params["doc_id"], ["CN118000001A"])
        self.assertEqual(params["client_sign"], ["abc"])

    def test_download_wire_params_and_path(self):
        fake = _FakeStreamClient()
        with patch("sources.baiten_client.httpx.AsyncClient",
                   return_value=fake):
            client = BaitenClient("k", "s", "http://gw")
            spool = asyncio.run(client.get_file("CN118000001A", "20240101"))
            spool.close()
        call = fake.calls[0]
        self.assertIn("/openService/download", call["url"])
        self.assertNotIn("/openService/file", call["url"])
        params = parse_qs(call["content"])
        self.assertEqual(params["pub_num"], ["CN118000001A"])
        self.assertEqual(params["pub_date"], ["20240101"])


class _FakeResponse:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self.text = text
        self._body = body

    def json(self):
        return self._body


class _FakeHttpClient:
    """httpx.AsyncClient double recording POST calls (context-managed)."""

    def __init__(self, status=200, body=None, text=""):
        self.calls = []
        self._status = status
        self._body = body
        self._text = text

    async def post(self, url, content, headers):
        self.calls.append({"url": url, "content": content, "headers": headers})
        return _FakeResponse(self._status, self._body, self._text)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeStreamResponse:
    status_code = 200

    def __init__(self, payload=b'{"fileByte": "AAEC"}'):
        self._chunks = [payload]

    async def aread(self):
        return b""

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeStreamClient:
    """httpx.AsyncClient double recording stream() calls (for get_file)."""

    def __init__(self):
        self.calls = []

    def stream(self, method, url, content, headers):
        # httpx's stream() is a plain method returning a context manager.
        self.calls.append({"url": url, "content": content, "headers": headers})
        return _FakeStreamResponse()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


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
