import unittest


class TestToolParamPolicy(unittest.TestCase):

    def test_backend_push_tools_are_exposed_even_when_query_and_body_are_empty(self):
        from sources.tool_param_policy import should_expose_dynamic_tool

        self.assertTrue(
            should_expose_dynamic_tool(
                push=2,
                has_tool_data=False,
                query_body_empty=True,
            )
        )

    def test_param_rules_allow_semantic_fill_inside_existing_empty_query_or_body(self):
        from sources.tool_param_policy import TEMPLATE_PARAM_RULES

        self.assertIn("may add or replace keys inside existing query/body objects", TEMPLATE_PARAM_RULES)
        self.assertNotIn("If a field is empty in the template, then leave it empty", TEMPLATE_PARAM_RULES)


if __name__ == "__main__":
    unittest.main()
