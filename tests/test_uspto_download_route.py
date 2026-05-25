import unittest


class TestUsptoDownloadRoute(unittest.TestCase):

    def test_fetches_uspto_download_response_as_file_payload(self):
        import sources.uspto_download as uspto_download

        class CaptureLogger:
            def __init__(self):
                self.messages = []

            def info(self, message):
                self.messages.append(message)

        class FakeResponse:
            status_code = 200
            content = b"%PDF-1.4 file bytes"
            headers = {
                "Content-Type": "application/pdf",
                "Content-Disposition": 'attachment; filename="document.pdf"',
            }

            def raise_for_status(self):
                return None

        calls = []
        capture_logger = CaptureLogger()
        original_logger = getattr(uspto_download, "logger", None)
        had_logger = hasattr(uspto_download, "logger")
        uspto_download.logger = capture_logger

        def restore_logger():
            if had_logger:
                uspto_download.logger = original_logger
            else:
                delattr(uspto_download, "logger")

        self.addCleanup(restore_logger)

        def fetch_response(url, headers):
            calls.append((url, headers))
            return FakeResponse()

        payload = uspto_download.fetch_uspto_download_file(
            "https://api.uspto.gov/api/v1/download/applications/18244278/documents/file.pdf",
            fetch_response=fetch_response,
            request_headers={"X-API-KEY": "server-secret"},
        )

        self.assertEqual(payload.content, b"%PDF-1.4 file bytes")
        self.assertEqual(payload.media_type, "application/pdf")
        self.assertEqual(payload.filename, "document.pdf")
        self.assertEqual(
            calls,
            [
                (
                    "https://api.uspto.gov/api/v1/download/applications/18244278/documents/file.pdf",
                    {"X-API-KEY": "server-secret"},
                )
            ]
        )
        self.assertIn("USPTO download response status: 200", capture_logger.messages)
        self.assertIn("USPTO download response content length: 19", capture_logger.messages)

    def test_rejects_non_uspto_download_url(self):
        from sources.uspto_download import fetch_uspto_download_file

        with self.assertRaisesRegex(ValueError, "Unsupported USPTO download URL"):
            fetch_uspto_download_file(
                "https://example.com/file.pdf",
                fetch_response=lambda url, headers: None,
                request_headers={},
            )

    def test_follows_text_response_url_to_file_payload(self):
        from sources.uspto_download import fetch_uspto_download_file

        class TextResponse:
            status_code = 200
            content = b'{"downloadUrl": "https://download.example.com/file.pdf"}'
            headers = {"Content-Type": "application/json"}

            def raise_for_status(self):
                return None

        class FileResponse:
            status_code = 200
            content = b"%PDF-1.4 resolved file bytes"
            headers = {
                "Content-Type": "application/pdf",
                "Content-Disposition": 'attachment; filename="resolved.pdf"',
            }

            def raise_for_status(self):
                return None

        calls = []

        def fetch_response(url, headers):
            calls.append(url)
            if len(calls) == 1:
                return TextResponse()
            return FileResponse()

        payload = fetch_uspto_download_file(
            "https://api.uspto.gov/api/v1/download/applications/18244278/documents/file.pdf",
            fetch_response=fetch_response,
            request_headers={},
        )

        self.assertEqual(
            calls,
            [
                "https://api.uspto.gov/api/v1/download/applications/18244278/documents/file.pdf",
                "https://download.example.com/file.pdf",
            ]
        )
        self.assertEqual(payload.content, b"%PDF-1.4 resolved file bytes")
        self.assertEqual(payload.media_type, "application/pdf")
        self.assertEqual(payload.filename, "resolved.pdf")

    def test_follows_uspto_redirect_instruction_even_when_octet_stream(self):
        from sources.uspto_download import fetch_uspto_download_file

        redirect_url = (
            "https://data-documents.uspto.gov/redirect/download/applications/"
            "18893954/MMU2X3JJX89X113.pdf?redirect_request_id="
            "436720a0-8b9f-4481-93e7-3b0f476aad0d"
        )

        class InstructionResponse:
            status_code = 200
            content = (
                "Please use redirect URL to downoload: "
                f"{redirect_url}. This URL is valid only for 3600 seconds."
            ).encode("utf-8")
            headers = {"Content-Type": "application/octet-stream"}

            def raise_for_status(self):
                return None

        class FileResponse:
            status_code = 200
            content = b"%PDF-1.4 redirected file bytes"
            headers = {"Content-Type": "application/pdf"}

            def raise_for_status(self):
                return None

        calls = []

        def fetch_response(url, headers):
            calls.append(url)
            if len(calls) == 1:
                return InstructionResponse()
            return FileResponse()

        payload = fetch_uspto_download_file(
            "https://api.uspto.gov/api/v1/download/applications/18893954/MMU2X3JJX89X113.pdf",
            fetch_response=fetch_response,
            request_headers={},
        )

        self.assertEqual(calls[1], redirect_url)
        self.assertEqual(payload.content, b"%PDF-1.4 redirected file bytes")
        self.assertEqual(payload.media_type, "application/pdf")
        self.assertEqual(payload.filename, "MMU2X3JJX89X113.pdf")

    def test_rejects_text_error_response_instead_of_downloading_it(self):
        from sources.uspto_download import fetch_uspto_download_file

        class ErrorResponse:
            status_code = 200
            content = b'{"error": "not found"}'
            headers = {"Content-Type": "application/json"}

            def raise_for_status(self):
                return None

        with self.assertRaisesRegex(ValueError, "did not contain downloadable file content"):
            fetch_uspto_download_file(
                "https://api.uspto.gov/api/v1/download/applications/18244278/documents/file.pdf",
                fetch_response=lambda url, headers: ErrorResponse(),
                request_headers={},
            )

    def test_rejects_plain_string_response_instead_of_downloading_empty_file(self):
        from sources.uspto_download import fetch_uspto_download_file

        with self.assertRaisesRegex(ValueError, "did not contain downloadable file content"):
            fetch_uspto_download_file(
                "https://api.uspto.gov/api/v1/download/applications/18893954/M1FWLUKIWFYBX92.pdf",
                fetch_response=lambda url, headers: '{"message":"Forbidden"}',
                request_headers={},
            )


if __name__ == "__main__":
    unittest.main()
