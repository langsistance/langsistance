import json
import unittest

from sources.result_pipeline import (
    ResultPipeline,
    find_primary_list,
    profile_items_schema,
)


class FakeCallback:
    def __init__(self):
        self.tokens = []

    async def on_llm_new_token(self, token):
        self.tokens.append(token)

    @property
    def text(self):
        return "".join(self.tokens)


class FakeLLM:
    def __init__(self, filter_payloads=None):
        self.filter_payloads = list(filter_payloads or [])
        self.filter_requests = []
        self.format_requests = []

    async def complete_simple(self, system_prompt, user_content):
        self.filter_requests.append((system_prompt, user_content))
        return self.filter_payloads.pop(0)

    async def stream_simple(self, system_prompt, user_content, callback_handler=None):
        self.format_requests.append((system_prompt, user_content))
        if callback_handler:
            await callback_handler.on_llm_new_token("FORMATTED\n")


class TestResultPipeline(unittest.IsolatedAsyncioTestCase):

    def test_find_primary_list_prefers_common_result_paths(self):
        data = {
            "ok": True,
            "data": {
                "rows": [
                    {"name": "A", "price": 10},
                    {"name": "B", "price": 20},
                ],
                "tags": ["small", "internal", "cached"],
            },
        }

        located = find_primary_list(data)

        self.assertEqual(located.path, "$.data.rows")
        self.assertEqual(len(located.items), 2)

    def test_profile_items_schema_infers_nested_paths_and_examples(self):
        profile = profile_items_schema([
            {
                "name": "A",
                "price": 10,
                "meta": {"available": True},
            },
            {
                "name": "B",
                "price": 20.5,
                "meta": {"available": False},
            },
        ])

        self.assertEqual(profile["fields"]["name"]["type"], "string")
        self.assertEqual(profile["fields"]["price"]["type"], "number")
        self.assertEqual(profile["fields"]["meta.available"]["type"], "boolean")
        self.assertEqual(profile["fields"]["name"]["examples"], ["A", "B"])

    async def test_streams_filter_progress_and_formats_only_matching_items(self):
        llm = FakeLLM([
            json.dumps({
                "matches": [
                    {"index": 0, "keep": False, "reason": "too expensive"},
                    {"index": 1, "keep": True, "reason": "fits"},
                ]
            }),
            json.dumps({
                "matches": [
                    {"index": 0, "keep": True, "reason": "verified"},
                ]
            }),
            json.dumps({
                "matches": [
                    {"index": 2, "keep": True, "reason": "fits"},
                ]
            }),
            json.dumps({
                "matches": [
                    {"index": 0, "keep": True, "reason": "verified"},
                ]
            }),
        ])
        callback = FakeCallback()
        pipeline = ResultPipeline(llm=llm, callback_handler=callback, batch_size=2)
        items = [
            {"name": "A", "price": 200},
            {"name": "B", "price": 80},
            {"name": "C", "price": 60},
        ]

        summary = await pipeline.stream_items(
            items,
            user_instruction="筛选价格低于 100 的商品",
            requires_filter=True,
        )

        self.assertEqual(summary.total, 3)
        self.assertEqual(summary.matched, 2)
        self.assertEqual(len(llm.filter_requests), 4)
        self.assertEqual(len(llm.format_requests), 2)
        self.assertIn("正在筛选 1-2 / 3", callback.text)
        self.assertIn("找到 1 条匹配项", callback.text)
        self.assertIn("筛选完成：共检查 3 条，匹配 2 条。", callback.text)
        self.assertIn('"name": "B"', llm.format_requests[0][1])
        self.assertIn('"name": "C"', llm.format_requests[1][1])

    async def test_streams_formatting_without_filter_when_no_filter_requested(self):
        llm = FakeLLM()
        callback = FakeCallback()
        pipeline = ResultPipeline(llm=llm, callback_handler=callback, batch_size=2)

        summary = await pipeline.stream_items(
            [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            user_instruction="列出这些结果",
            requires_filter=False,
        )

        self.assertEqual(summary.total, 3)
        self.assertEqual(summary.matched, 3)
        self.assertEqual(len(llm.filter_requests), 0)
        self.assertEqual(len(llm.format_requests), 2)
        self.assertIn("已获取 3 条结果，正在整理输出", callback.text)


    async def test_filter_verifier_rejects_candidates_that_do_not_strictly_match_instruction(self):
        llm = FakeLLM([
            json.dumps({
                "matches": [
                    {"index": 0, "keep": True, "reason": "company"},
                    {"index": 1, "keep": True, "reason": "incorrectly kept individual"},
                ]
            }),
            json.dumps({
                "matches": [
                    {"index": 0, "keep": True, "reason": "assignor is a company"},
                    {"index": 1, "keep": False, "reason": "assignor is an individual"},
                ]
            })
        ])
        callback = FakeCallback()
        pipeline = ResultPipeline(llm=llm, callback_handler=callback, batch_size=2)
        items = [
            {
                "title": "Company-originated patent",
                "assignee": "Samsung Display Co., Ltd.",
                "assignors": ["Samsung Display Co., Ltd."],
            },
            {
                "title": "Individual-originated patent",
                "assignee": "Samsung Display Co., Ltd.",
                "assignors": ["KIM, JIN SOO"],
            },
        ]

        summary = await pipeline.stream_items(
            items,
            user_instruction=(
                "Search for patents with Samsung Display Co., Ltd. as the patent assignee. "
                "I only want patents where the assignors are companies, not individual."
            ),
            requires_filter=True,
        )

        self.assertEqual(summary.total, 2)
        self.assertEqual(summary.matched, 1)
        self.assertEqual(len(llm.format_requests), 1)
        formatted_payload = llm.format_requests[0][1]
        self.assertIn("Company-originated patent", formatted_payload)
        self.assertNotIn("Individual-originated patent", formatted_payload)


if __name__ == "__main__":
    unittest.main()
