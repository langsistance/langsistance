import unittest


class TestDynamicToolParams(unittest.TestCase):

    def test_accepts_structured_params_without_json_decoding(self):
        from sources.dynamic_tool_params import _coerce_json_object

        params = {
            "query": {},
            "body": {
                "q": 'applicationNumberText:"18244278"'
            }
        }

        self.assertIs(_coerce_json_object(params, "LLM tool params"), params)

    def test_accepts_legacy_json_string_params(self):
        from sources.dynamic_tool_params import _coerce_json_object

        params = '{"query": {}, "body": {"q": "applicationNumberText:\\"18244278\\""}}'

        parsed = _coerce_json_object(params, "LLM tool params")

        self.assertEqual(parsed["body"]["q"], 'applicationNumberText:"18244278"')

    def test_reports_invalid_llm_tool_params_separately(self):
        from sources.dynamic_tool_params import _coerce_json_object

        params = '{"query": {}, "body": {"q": "applicationNumberText:\\"18244278\\""}}}'

        with self.assertRaisesRegex(ValueError, "Invalid JSON in LLM tool params"):
            _coerce_json_object(params, "LLM tool params")

    def test_repairs_missing_comma_between_string_fields(self):
        from sources.dynamic_tool_params import _coerce_json_object

        # Simulates LLM forgetting the comma between "value" and next "key"
        params = '{"query": {"q": "apple""assignee": "Google"}}'
        result = _coerce_json_object(params, "LLM tool params")
        self.assertEqual(result["query"]["q"], "apple")
        self.assertEqual(result["query"]["assignee"], "Google")

    def test_repairs_trailing_comma_in_object(self):
        from sources.dynamic_tool_params import _coerce_json_object

        params = '{"query": {"q": "apple",}, "body": {}}'
        result = _coerce_json_object(params, "LLM tool params")
        self.assertEqual(result["query"]["q"], "apple")

    def test_repairs_trailing_comma_in_array(self):
        from sources.dynamic_tool_params import _coerce_json_object

        params = '{"items": ["a", "b", "c",]}'
        result = _coerce_json_object(params, "LLM tool params")
        self.assertEqual(result["items"], ["a", "b", "c"])

    def test_repairs_missing_comma_after_closing_brace(self):
        from sources.dynamic_tool_params import _coerce_json_object

        # Missing comma between }"next_key":
        params = '{"body": {"q": "search"}"query": {"page": 1}}'
        result = _coerce_json_object(params, "LLM tool params")
        self.assertEqual(result["body"]["q"], "search")
        self.assertEqual(result["query"]["page"], 1)

    def test_repairs_single_quoted_keys(self):
        from sources.dynamic_tool_params import _coerce_json_object

        params = "{'query': {'q': 'apple'}, 'body': {}}"
        result = _coerce_json_object(params, "LLM tool params")
        self.assertEqual(result["query"]["q"], "apple")

    def test_repair_chain_multiple_fixes(self):
        from sources.dynamic_tool_params import _coerce_json_object

        # Missing comma + trailing comma
        params = '{"query": {"q": "test""assignee": "Apple",}, "body": {}}'
        result = _coerce_json_object(params, "LLM tool params")
        self.assertEqual(result["query"]["q"], "test")
        self.assertEqual(result["query"]["assignee"], "Apple")

    def test_logs_warning_and_raw_json_on_repair(self):
        import sources.dynamic_tool_params as dynamic_tool_params

        class CaptureLogger:
            def __init__(self):
                self.messages = []

            def info(self, message):
                self.messages.append(("info", message))

            def warning(self, message):
                self.messages.append(("warning", message))

        capture_logger = CaptureLogger()
        original_logger = getattr(dynamic_tool_params, "logger", None)
        dynamic_tool_params.logger = capture_logger

        def restore_logger():
            dynamic_tool_params.logger = original_logger

        self.addCleanup(restore_logger)

        # Malformed but repairable JSON
        params = '{"query": {"q": "apple",}, "body": {}}'
        result = dynamic_tool_params._coerce_json_object(params, "LLM tool params")

        self.assertEqual(result["query"]["q"], "apple")

        # Check that a warning was logged about the invalid JSON
        warning_messages = [m for level, m in capture_logger.messages if level == "warning"]
        self.assertTrue(any("LLM produced invalid JSON" in m for m in warning_messages))
        self.assertTrue(any("Raw JSON preview" in m for m in warning_messages))

        # Check that repair success was logged
        info_messages = [m for level, m in capture_logger.messages if level == "info"]
        self.assertTrue(any("Successfully repaired JSON" in m for m in info_messages))

    def test_appends_path_to_base_url(self):
        from sources.dynamic_tool_params import _append_path_to_url

        url = _append_path_to_url(
            "https://api.example.com/v1/applications",
            "/18244278/document"
        )

        self.assertEqual(
            url,
            "https://api.example.com/v1/applications/18244278/document"
        )

    def test_rejects_absolute_path_url(self):
        from sources.dynamic_tool_params import _append_path_to_url

        with self.assertRaisesRegex(ValueError, "path must be a URL path"):
            _append_path_to_url(
                "https://api.example.com/v1/applications",
                "https://evil.example/document"
            )

    def test_treats_missing_or_empty_path_query_body_as_empty(self):
        from sources.dynamic_tool_params import _is_path_query_body_empty

        self.assertTrue(_is_path_query_body_empty({
            "header": {
                "X-API-KEY": "server-secret"
            }
        }))
        self.assertTrue(_is_path_query_body_empty({
            "path": "",
            "query": {},
            "body": None,
            "header": {
                "X-API-KEY": "server-secret"
            }
        }))

    def test_non_empty_path_requires_llm_tool_params(self):
        from sources.dynamic_tool_params import _is_path_query_body_empty

        self.assertFalse(_is_path_query_body_empty({
            "path": "/{applicationNumberText}/document",
            "query": {},
            "body": {}
        }))

    def test_replaces_uspto_download_urls_with_proxy_urls_without_fetching(self):
        from sources.dynamic_tool_params import _replace_uspto_download_urls

        calls = []

        def fetch_text(url, headers):
            calls.append((url, headers))
            return 'download link: "https://download.example.com/file.pdf"'

        result_data = {
            "documentBag": [
                {
                    "downloadOptionBag": [
                        {
                            "downloadUrl": (
                                "https://api.uspto.gov/api/v1/download/applications/"
                                "18244278/documents/example.pdf"
                            )
                        },
                        {
                            "downloadUrl": "https://already.example.com/file.pdf"
                        }
                    ]
                }
            ]
        }
        headers = {"X-API-KEY": "server-secret"}

        updated = _replace_uspto_download_urls(
            result_data,
            proxy_base_url="https://api.copiioai.com"
        )

        self.assertIs(updated, result_data)
        self.assertEqual(calls, [])
        rewritten_url = result_data["documentBag"][0]["downloadOptionBag"][0]["downloadUrl"]
        self.assertTrue(
            rewritten_url.startswith("https://api.copiioai.com/uspto/download?url=")
        )
        self.assertIn(
            "https%3A%2F%2Fapi.uspto.gov%2Fapi%2Fv1%2Fdownload%2Fapplications%2F",
            rewritten_url
        )
        self.assertIn(
            "18244278%2Fdocuments%2Fexample.pdf",
            rewritten_url
        )
        self.assertEqual(
            result_data["documentBag"][0]["downloadOptionBag"][1]["downloadUrl"],
            "https://already.example.com/file.pdf"
        )

    def test_replaces_uspto_download_urls_inside_list_items(self):
        from sources.dynamic_tool_params import _replace_uspto_download_urls

        result_data = [
            {
                "documentBag": [
                    {
                        "downloadOptionBag": [
                            {
                                "downloadUrl": (
                                    "https://api.uspto.gov/api/v1/download/applications/"
                                    "18244278/documents/example.pdf"
                                )
                            }
                        ]
                    }
                ]
            }
        ]

        _replace_uspto_download_urls(result_data, proxy_base_url="https://api.copiioai.com")

        self.assertTrue(
            result_data[0]["documentBag"][0]["downloadOptionBag"][0]["downloadUrl"]
            .startswith("https://api.copiioai.com/uspto/download?url=")
        )

    def test_replaces_uspto_download_urls_when_batch_is_document_bag(self):
        from sources.dynamic_tool_params import _replace_uspto_download_urls_for_batch

        batch = [
            {
                "downloadOptionBag": [
                    {
                        "downloadUrl": (
                            "https://api.uspto.gov/api/v1/download/applications/"
                            "18244278/documents/direct.pdf"
                        )
                    }
                ]
            }
        ]

        _replace_uspto_download_urls_for_batch(
            batch,
            proxy_base_url="https://api.copiioai.com"
        )

        self.assertTrue(
            batch[0]["downloadOptionBag"][0]["downloadUrl"]
            .startswith("https://api.copiioai.com/uspto/download?url=")
        )

    def test_logs_uspto_document_bags_options_and_url_replacements(self):
        import sources.dynamic_tool_params as dynamic_tool_params

        class CaptureLogger:
            def __init__(self):
                self.messages = []

            def info(self, message):
                self.messages.append(message)

        capture_logger = CaptureLogger()
        original_logger = getattr(dynamic_tool_params, "logger", None)
        had_logger = hasattr(dynamic_tool_params, "logger")
        dynamic_tool_params.logger = capture_logger

        def restore_logger():
            if had_logger:
                dynamic_tool_params.logger = original_logger
            else:
                delattr(dynamic_tool_params, "logger")

        self.addCleanup(restore_logger)

        original_url = (
            "https://api.uspto.gov/api/v1/download/applications/"
            "18244278/documents/logged.pdf"
        )
        result_data = {
            "documentBag": [
                {
                    "downloadOptionBag": [
                        {"downloadUrl": original_url}
                    ]
                }
            ]
        }

        dynamic_tool_params._replace_uspto_download_urls(
            result_data,
            proxy_base_url="https://api.copiioai.com"
        )

        log_text = "\n".join(capture_logger.messages)
        self.assertIn("documentBag:", log_text)
        self.assertIn("downloadOptionBag:", log_text)
        self.assertIn(f"downloadUrl: {original_url}", log_text)
        self.assertIn(
            "replaced downloadUrl: https://api.copiioai.com/uspto/download?url=",
            log_text
        )

    def test_resolves_only_current_result_batch(self):
        from sources.dynamic_tool_params import _replace_uspto_download_urls_for_batch

        def make_item(index):
            return {
                "documentBag": [
                    {
                        "downloadOptionBag": [
                            {
                                "downloadUrl": (
                                    "https://api.uspto.gov/api/v1/download/applications/"
                                    f"1824427{index}/documents/example.pdf"
                                )
                            }
                        ]
                    }
                ]
            }

        pending = [make_item(i) for i in range(6)]
        calls = []

        def fetch_text(url, headers):
            calls.append(url)
            return f"https://download.example.com/{len(calls)}.pdf"

        _replace_uspto_download_urls_for_batch(
            pending[:5],
            proxy_base_url="https://api.copiioai.com"
        )

        self.assertEqual(calls, [])
        self.assertTrue(
            pending[0]["documentBag"][0]["downloadOptionBag"][0]["downloadUrl"]
            .startswith("https://api.copiioai.com/uspto/download?url=")
        )
        self.assertTrue(
            pending[5]["documentBag"][0]["downloadOptionBag"][0]["downloadUrl"]
            .startswith("https://api.uspto.gov/api/v1/download/applications")
        )


if __name__ == "__main__":
    unittest.main()
