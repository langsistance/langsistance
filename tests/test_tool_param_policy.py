import unittest


class TestToolParamPolicy(unittest.TestCase):

    def test_empty_query_and_body_ignore_llm_generated_params(self):
        from sources.tool_param_policy import normalize_tool_request_params

        template = {
            "method": "GET",
            "query": {},
            "body": {},
        }
        llm_params = {
            "query": {"q": "Samsung Display Co., Ltd."},
            "body": {"assignee": "Samsung Display Co., Ltd."},
        }

        normalized = normalize_tool_request_params(template, llm_params)

        self.assertEqual(normalized["query"], {})
        self.assertEqual(normalized["body"], {})

    def test_non_empty_query_and_body_allow_replace_and_delete_but_not_add(self):
        from sources.tool_param_policy import normalize_tool_request_params

        template = {
            "method": "POST",
            "query": {"q": "", "page": "1"},
            "body": {"assignee": "", "status": "active"},
        }
        llm_params = {
            "query": {"q": "Samsung Display Co., Ltd.", "extra": "invented"},
            "body": {"assignee": "Samsung Display Co., Ltd.", "newField": "invented"},
        }

        normalized = normalize_tool_request_params(template, llm_params)

        self.assertEqual(normalized["query"], {"q": "Samsung Display Co., Ltd."})
        self.assertEqual(normalized["body"], {"assignee": "Samsung Display Co., Ltd."})
        self.assertNotIn("extra", normalized["query"])
        self.assertNotIn("newField", normalized["body"])

    def test_prompt_rules_match_param_policy(self):
        from sources.tool_param_policy import TEMPLATE_PARAM_RULES

        self.assertIn("If query or body is empty in the template, it MUST remain empty", TEMPLATE_PARAM_RULES)
        self.assertIn("may replace or remove existing keys", TEMPLATE_PARAM_RULES)
        self.assertIn("MUST NOT add new keys", TEMPLATE_PARAM_RULES)
        self.assertIn("narrow the existing expression", TEMPLATE_PARAM_RULES)
        self.assertIn("OR alternatives", TEMPLATE_PARAM_RULES)

    def test_empty_query_and_body_use_direct_tool_call_without_llm_param_generation(self):
        from sources.tool_param_policy import should_expose_dynamic_tool, should_prefetch_tool_result

        self.assertFalse(
            should_expose_dynamic_tool(
                has_tool_data=False,
                query_body_empty=True,
            )
        )
        self.assertTrue(
            should_prefetch_tool_result(
                has_tool_data=False,
                query_body_empty=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
