import unittest
import sys
import threading
import time
import types
from unittest.mock import patch

if "requests" not in sys.modules:
    requests_module = types.ModuleType("requests")
    requests_module.request = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_module

from sources.http_outbound import (
    OutboundHttpBlockedError,
    OutboundHttpConcurrencyTimeoutError,
    outbound_http,
    validate_outbound_url,
)


class TestHttpOutbound(unittest.TestCase):
    def test_validate_outbound_url_rejects_localhost(self):
        with self.assertRaises(OutboundHttpBlockedError):
            validate_outbound_url("http://localhost:8080/search")

    def test_validate_outbound_url_rejects_loopback_ip(self):
        with self.assertRaises(OutboundHttpBlockedError):
            validate_outbound_url("http://127.0.0.1:8080/search")

    def test_outbound_request_preserves_tool_request_shape(self):
        captured = {}

        class FakeResponse:
            status_code = 200
            headers = {"Content-Type": "application/json"}
            content = b'{"ok": true}'
            text = '{"ok": true}'

            def json(self):
                return {"ok": True}

        def fake_request(method, url, **kwargs):
            captured["method"] = method
            captured["url"] = url
            captured["kwargs"] = kwargs
            return FakeResponse()

        with patch("sources.http_outbound.requests.request", fake_request):
            response = outbound_http.request(
                "POST",
                "https://api.example.com/items",
                purpose="backend_tool",
                params={"q": "patent"},
                headers={"Authorization": "Bearer token"},
                json={"limit": 10},
                timeout=17,
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured, {
            "method": "POST",
            "url": "https://api.example.com/items",
            "kwargs": {
                "params": {"q": "patent"},
                "headers": {"Authorization": "Bearer token"},
                "json": {"limit": 10},
                "timeout": 17,
            },
        })

    def test_api_uspto_gov_allows_only_one_in_flight_request(self):
        first_request_started = threading.Event()
        release_first_request = threading.Event()

        class FakeResponse:
            status_code = 200

        def fake_request(method, url, **kwargs):
            if url.endswith("/first"):
                first_request_started.set()
                release_first_request.wait(timeout=3)
            return FakeResponse()

        with patch("sources.http_outbound.requests.request", fake_request):
            first_thread = threading.Thread(
                target=lambda: outbound_http.get("https://api.uspto.gov/first", purpose="download")
            )
            first_thread.start()
            self.assertTrue(first_request_started.wait(timeout=1))

            started_at = time.monotonic()
            with self.assertRaises(OutboundHttpConcurrencyTimeoutError):
                outbound_http.get("https://api.uspto.gov/second", purpose="download")
            elapsed = time.monotonic() - started_at

            release_first_request.set()
            first_thread.join(timeout=1)

        self.assertGreaterEqual(elapsed, 1)
        self.assertLess(elapsed, 2)


if __name__ == "__main__":
    unittest.main()
