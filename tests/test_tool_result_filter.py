import unittest


class TestToolResultFilter(unittest.IsolatedAsyncioTestCase):

    async def test_filters_items_using_high_confidence_keep_decisions(self):
        from sources.tool_result_filter import filter_tool_result_items

        items = [
            {"name": "Alpha", "city": "Tokyo", "url": "https://example.com/a"},
            {"name": "Beta", "city": "Osaka", "url": "https://example.com/b"},
            {"name": "Gamma", "city": "Tokyo", "url": "https://example.com/c"},
        ]

        async def llm_json_call(system_prompt, user_content):
            return {
                "has_filter_requirement": True,
                "decisions": [
                    {"index": 0, "keep": True, "confidence": 0.96, "reason": "Tokyo"},
                    {"index": 1, "keep": False, "confidence": 0.91, "reason": "Osaka"},
                    {"index": 2, "keep": True, "confidence": 0.93, "reason": "Tokyo"},
                ],
            }

        result = await filter_tool_result_items(
            items,
            "只保留东京的公司",
            llm_json_call,
        )

        self.assertTrue(result.applied)
        self.assertEqual(result.original_count, 3)
        self.assertEqual(result.filtered_count, 2)
        self.assertEqual(result.items, [items[0], items[2]])

    async def test_keeps_all_items_when_no_filter_requirement(self):
        from sources.tool_result_filter import filter_tool_result_items

        items = [{"name": "Alpha"}, {"name": "Beta"}]

        async def llm_json_call(system_prompt, user_content):
            return {
                "has_filter_requirement": False,
                "decisions": [
                    {"index": 0, "keep": True, "confidence": 1.0},
                    {"index": 1, "keep": True, "confidence": 1.0},
                ],
            }

        result = await filter_tool_result_items(
            items,
            "列出这些结果",
            llm_json_call,
        )

        self.assertFalse(result.applied)
        self.assertEqual(result.items, items)

    async def test_missing_and_low_confidence_reject_decisions_keep_items(self):
        from sources.tool_result_filter import filter_tool_result_items

        items = [{"name": "Alpha"}, {"name": "Beta"}, {"name": "Gamma"}]

        async def llm_json_call(system_prompt, user_content):
            return {
                "has_filter_requirement": True,
                "decisions": [
                    {"index": 0, "keep": False, "confidence": 0.95},
                    {"index": 2, "keep": False, "confidence": 0.42},
                ],
            }

        result = await filter_tool_result_items(
            items,
            "过滤掉不符合条件的项目",
            llm_json_call,
        )

        self.assertTrue(result.applied)
        self.assertEqual(result.items, [items[1], items[2]])

    async def test_invalid_llm_json_keeps_original_list(self):
        from sources.tool_result_filter import filter_tool_result_items

        items = [{"name": "Alpha"}, {"name": "Beta"}]

        async def llm_json_call(system_prompt, user_content):
            return "not json"

        result = await filter_tool_result_items(
            items,
            "只保留符合条件的项目",
            llm_json_call,
        )

        self.assertFalse(result.applied)
        self.assertEqual(result.items, items)
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()
