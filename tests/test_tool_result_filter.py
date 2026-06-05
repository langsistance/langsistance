import unittest


class CaptureLogger:
    def __init__(self):
        self.infos = []
        self.errors = []

    def info(self, message):
        self.infos.append(message)

    def error(self, message):
        self.errors.append(message)


class TestToolResultFilter(unittest.IsolatedAsyncioTestCase):

    def test_filter_prompt_prevents_related_property_substitution_and_inconsistent_decisions(self):
        from sources.tool_result_filter import FILTER_SYSTEM_PROMPT

        prompt = " ".join(FILTER_SYSTEM_PROMPT.lower().split())

        self.assertIn("keep value and reason must agree", prompt)
        self.assertIn("exact item field, entity role, attribute, or property named in the filter criteria", prompt)
        self.assertIn("different related field", prompt)
        self.assertIn("core search goal", prompt)
        self.assertIn("any member of a collection", prompt)
        self.assertIn("all members do not match", prompt)
        self.assertIn("set keep to false", prompt)

    async def test_filters_items_using_high_confidence_keep_decisions(self):
        from sources.tool_result_filter import filter_tool_result_items

        items = [
            {"name": "Alpha", "city": "Tokyo", "url": "https://example.com/a"},
            {"name": "Beta", "city": "Osaka", "url": "https://example.com/b"},
            {"name": "Gamma", "city": "Tokyo", "url": "https://example.com/c"},
        ]
        llm_calls = []
        responses = [
            {
                "has_filter_criteria": True,
                "filter_criteria": "Tokyo results",
            },
            {
                "has_filter_requirement": True,
                "decisions": [
                    {"index": 0, "keep": True, "confidence": 0.96, "reason": "Tokyo"},
                    {"index": 1, "keep": False, "confidence": 0.91, "reason": "Osaka"},
                    {"index": 2, "keep": True, "confidence": 0.93, "reason": "Tokyo"},
                ],
            },
        ]

        async def llm_json_call(system_prompt, user_content):
            llm_calls.append({"system_prompt": system_prompt, "user_content": user_content})
            return responses[len(llm_calls) - 1]

        result = await filter_tool_result_items(
            items,
            "Please show these companies, but only keep Tokyo results.",
            llm_json_call,
            batch_size=len(items),
        )

        self.assertTrue(result.applied)
        self.assertEqual(result.original_count, 3)
        self.assertEqual(result.filtered_count, 2)
        self.assertEqual(result.items, [items[0], items[2]])
        self.assertEqual(len(llm_calls), 2)
        self.assertIn("Filter criteria:\nTokyo results", llm_calls[1]["user_content"])

    async def test_keeps_all_items_when_no_filter_requirement(self):
        from sources.tool_result_filter import filter_tool_result_items

        items = [{"name": "Alpha"}, {"name": "Beta"}]
        llm_calls = []

        async def llm_json_call(system_prompt, user_content):
            llm_calls.append(user_content)
            return {
                "has_filter_criteria": False,
                "filter_criteria": "",
            }

        result = await filter_tool_result_items(
            items,
            "List these results.",
            llm_json_call,
        )

        self.assertFalse(result.applied)
        self.assertEqual(result.items, items)
        self.assertEqual(len(llm_calls), 1)

    async def test_criteria_prompt_distinguishes_core_goal_from_supplemental_filter(self):
        from sources.tool_result_filter import filter_tool_result_items

        items = [{"name": "Alpha"}]
        llm_calls = []

        async def llm_json_call(system_prompt, user_content):
            llm_calls.append({
                "system_prompt": system_prompt,
                "user_content": user_content,
            })
            return {
                "has_filter_criteria": False,
                "filter_criteria": "",
            }

        await filter_tool_result_items(
            items,
            "Find patent documents for application 18893954.",
            llm_json_call,
        )

        self.assertEqual(len(llm_calls), 1)
        criteria_prompt = llm_calls[0]["system_prompt"]
        self.assertIn("core goal", criteria_prompt)
        self.assertIn("supplemental condition", criteria_prompt)
        self.assertIn("single core goal", criteria_prompt)
        self.assertIn("not a result filter", criteria_prompt)

    async def test_criteria_prompt_treats_single_retrieval_constraint_as_core_goal(self):
        from sources.tool_result_filter import filter_tool_result_items

        items = [{"name": "Alpha"}]
        llm_calls = []

        async def llm_json_call(system_prompt, user_content):
            llm_calls.append({
                "system_prompt": system_prompt,
                "user_content": user_content,
            })
            return {
                "has_filter_criteria": False,
                "filter_criteria": "",
            }

        await filter_tool_result_items(
            items,
            "Search for records with Acme Corp as the owner.",
            llm_json_call,
        )

        self.assertEqual(len(llm_calls), 1)
        criteria_prompt = llm_calls[0]["system_prompt"]
        self.assertIn("first or only constraint", criteria_prompt)
        self.assertIn("used to retrieve the core result set", criteria_prompt)
        self.assertIn("not a result filter", criteria_prompt)

    async def test_criteria_prompt_allows_supplemental_attribute_constraints_as_filters(self):
        from sources.tool_result_filter import filter_tool_result_items

        items = [{"name": "Alpha"}]
        llm_calls = []

        async def llm_json_call(system_prompt, user_content):
            llm_calls.append({
                "system_prompt": system_prompt,
                "user_content": user_content,
            })
            return {
                "has_filter_criteria": True,
                "filter_criteria": "patentee is an organization rather than a person",
            }

        await filter_tool_result_items(
            items,
            (
                "Search for patents with Samsung Display Co., Ltd. as the patent assignee. "
                "I only want patents whose patentee is a corporation rather than a natural person."
            ),
            llm_json_call,
        )

        self.assertGreaterEqual(len(llm_calls), 1)
        criteria_prompt = llm_calls[0]["system_prompt"]
        self.assertIn("supplemental", criteria_prompt)
        self.assertIn("attribute", criteria_prompt)
        self.assertIn("category", criteria_prompt)
        self.assertIn("entity type", criteria_prompt)
        self.assertIn("not a natural person", criteria_prompt)

    async def test_missing_and_low_confidence_reject_decisions_keep_items(self):
        from sources.tool_result_filter import filter_tool_result_items

        items = [{"name": "Alpha"}, {"name": "Beta"}, {"name": "Gamma"}]
        responses = [
            {
                "has_filter_criteria": True,
                "filter_criteria": "items that match the condition",
            },
            {
                "has_filter_requirement": True,
                "decisions": [
                    {"index": 0, "keep": False, "confidence": 0.95},
                    {"index": 2, "keep": False, "confidence": 0.42},
                ],
            },
        ]
        call_count = 0

        async def llm_json_call(system_prompt, user_content):
            nonlocal call_count
            call_count += 1
            return responses[call_count - 1]

        result = await filter_tool_result_items(
            items,
            "Filter out items that do not match the condition.",
            llm_json_call,
            batch_size=len(items),
        )

        self.assertTrue(result.applied)
        self.assertEqual(result.items, [items[1], items[2]])

    async def test_emits_transient_status_while_filtering_batches(self):
        from sources.tool_result_filter import filter_tool_result_items

        items = [
            {"name": "Alpha"},
            {"name": "Beta"},
            {"name": "Gamma"},
        ]
        responses = [
            {
                "has_filter_criteria": True,
                "filter_criteria": "matching items",
            },
            {
                "has_filter_requirement": True,
                "decisions": [
                    {"index": 0, "keep": True, "confidence": 0.99},
                    {"index": 1, "keep": False, "confidence": 0.99},
                ],
            },
            {
                "has_filter_requirement": True,
                "decisions": [
                    {"index": 2, "keep": True, "confidence": 0.99},
                ],
            },
        ]
        statuses = []
        call_count = 0

        async def llm_json_call(system_prompt, user_content):
            nonlocal call_count
            call_count += 1
            return responses[call_count - 1]

        async def status_callback(event):
            statuses.append(event)

        result = await filter_tool_result_items(
            items,
            "Only keep matching items.",
            llm_json_call,
            batch_size=2,
            status_callback=status_callback,
        )

        self.assertTrue(result.applied)
        self.assertEqual(result.items, [items[0], items[2]])
        self.assertGreaterEqual(len(statuses), 4)
        self.assertEqual(statuses[0]["phase"], "criteria")
        self.assertEqual(statuses[0]["transient"], True)
        self.assertEqual(statuses[1]["phase"], "batch")
        self.assertEqual(statuses[1]["current"], 1)
        self.assertEqual(statuses[1]["end"], 2)
        self.assertEqual(statuses[1]["total"], 3)
        self.assertIn("Filtering results 1-2 of 3", statuses[1]["message"])
        self.assertEqual(statuses[-1]["phase"], "complete")
        self.assertEqual(statuses[-1]["kept"], 2)
        self.assertEqual(statuses[-1]["total"], 3)

    async def test_invalid_llm_json_keeps_original_list(self):
        from sources.tool_result_filter import filter_tool_result_items

        items = [{"name": "Alpha"}, {"name": "Beta"}]

        async def llm_json_call(system_prompt, user_content):
            if "Return this JSON shape exactly" in user_content:
                return "not json"
            return {
                "has_filter_criteria": True,
                "filter_criteria": "matching items",
            }

        result = await filter_tool_result_items(
            items,
            "Only keep matching items.",
            llm_json_call,
        )

        self.assertFalse(result.applied)
        self.assertEqual(result.items, items)
        self.assertIsNotNone(result.error)

    async def test_logs_filter_inputs_criteria_decisions_and_results(self):
        from sources import tool_result_filter

        capture_logger = CaptureLogger()
        original_logger = tool_result_filter.logger
        tool_result_filter.logger = capture_logger
        self.addCleanup(setattr, tool_result_filter, "logger", original_logger)

        items = [{"name": "Alpha"}, {"name": "Beta"}]
        responses = [
            {
                "has_filter_criteria": True,
                "filter_criteria": "Alpha",
            },
            {
                "has_filter_requirement": True,
                "decisions": [
                    {"index": 0, "keep": True, "confidence": 0.99, "reason": "matches"},
                    {"index": 1, "keep": False, "confidence": 0.94, "reason": "does not match"},
                ],
            },
        ]
        call_count = 0

        async def llm_json_call(system_prompt, user_content):
            nonlocal call_count
            call_count += 1
            return responses[call_count - 1]

        result = await tool_result_filter.filter_tool_result_items(
            items,
            "only keep Alpha",
            llm_json_call,
            batch_size=len(items),
        )

        info_log = "\n".join(capture_logger.infos)
        self.assertEqual(result.items, [items[0]])
        self.assertIn("filter input items", info_log)
        self.assertIn("filter input items (2)", info_log)
        self.assertIn("Alpha", info_log)
        self.assertIn("filter criteria", info_log)
        self.assertIn("filter criteria/keywords: Alpha", info_log)
        self.assertIn("filter decisions", info_log)
        self.assertIn("does not match", info_log)
        self.assertIn("filtered result items", info_log)
        self.assertIn("filtered result items (1)", capture_logger.infos[-1])

    async def test_filter_prompt_uses_extracted_filter_criteria_not_full_user_question(self):
        from sources.tool_result_filter import filter_tool_result_items

        items = [{"name": "Alpha", "publication": "US20250014493A1"}]
        captured_user_content = []
        responses = [
            {
                "has_filter_criteria": True,
                "filter_criteria": "patent publication number US20250014493A1",
            },
            {
                "has_filter_requirement": True,
                "decisions": [
                    {"index": 0, "keep": True, "confidence": 0.99},
                ],
            },
        ]

        async def llm_json_call(system_prompt, user_content):
            captured_user_content.append(user_content)
            return responses[len(captured_user_content) - 1]

        await filter_tool_result_items(
            items,
            "I want to obtain all patent documents with the patent publication number US20250014493A1.",
            llm_json_call,
        )

        self.assertEqual(len(captured_user_content), 2)
        self.assertIn("Filter criteria:", captured_user_content[1])
        self.assertIn("patent publication number US20250014493A1", captured_user_content[1])
        self.assertNotIn("I want to obtain all patent documents", captured_user_content[1])

    async def test_logs_fail_open_filter_error(self):
        from sources import tool_result_filter

        capture_logger = CaptureLogger()
        original_logger = tool_result_filter.logger
        tool_result_filter.logger = capture_logger
        self.addCleanup(setattr, tool_result_filter, "logger", original_logger)

        items = [{"name": "Alpha"}]

        async def llm_json_call(system_prompt, user_content):
            if "Return this JSON shape exactly" in user_content:
                return "not json"
            return {
                "has_filter_criteria": True,
                "filter_criteria": "Alpha",
            }

        result = await tool_result_filter.filter_tool_result_items(
            items,
            "only keep Alpha",
            llm_json_call,
        )

        error_log = "\n".join(capture_logger.errors)
        self.assertEqual(result.items, items)
        self.assertIn("filter failed open", error_log)
        self.assertIn("not json", error_log)
        self.assertIn("original items kept", error_log)


if __name__ == "__main__":
    unittest.main()
