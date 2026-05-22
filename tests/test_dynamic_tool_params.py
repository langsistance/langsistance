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


if __name__ == "__main__":
    unittest.main()
