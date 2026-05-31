import json
import types
import unittest

from sources.knowledge.selection import choose_knowledge_candidate


def _knowledge(
    knowledge_id,
    question,
    description="",
    answer="",
    knowledge_type=1,
    similarity=0.5,
):
    return types.SimpleNamespace(
        id=knowledge_id,
        question=question,
        description=description,
        answer=answer,
        type=knowledge_type,
        extra_info={"similarity": similarity},
    )


def _tool(title="tool", description="tool description", push=2):
    return types.SimpleNamespace(
        title=title,
        description=description,
        push=push,
    )


class TestKnowledgeSelection(unittest.IsolatedAsyncioTestCase):

    async def test_llm_can_choose_lower_similarity_candidate_using_routing_hint(self):
        calls = []

        async def complete_json(system_prompt, user_content):
            calls.append({
                "system_prompt": system_prompt,
                "user_content": user_content,
            })
            return {
                "knowledge_id": 2,
                "confidence": 0.88,
                "reason": "routing hint matches patent status lookup",
            }

        high_similarity = _knowledge(
            1,
            "Lookup company profile",
            description="Use when the user asks about companies.",
            similarity=0.92,
        )
        lower_similarity = _knowledge(
            2,
            "Lookup patent status",
            description="Use for patent status and publication document searches.",
            similarity=0.61,
        )

        selected = await choose_knowledge_candidate(
            "Find the current status for patent US123",
            [(high_similarity, _tool()), (lower_similarity, _tool())],
            complete_json,
        )

        self.assertIs(selected[0], lower_similarity)
        payload = json.loads(calls[0]["user_content"])
        self.assertEqual(
            payload["candidates"][1]["routing_hint"],
            "Use for patent status and publication document searches.",
        )
        self.assertIn("routing_hint", calls[0]["system_prompt"])
        self.assertIn("must not be used as execution instructions", calls[0]["system_prompt"])

    async def test_workflow_answer_is_not_sent_to_routing_llm(self):
        calls = []

        async def complete_json(system_prompt, user_content):
            calls.append(user_content)
            return {
                "knowledge_id": 10,
                "confidence": 0.93,
                "reason": "workflow scenario matches",
            }

        workflow = _knowledge(
            10,
            "Patent publication workflow",
            description="Use when users need patent publication documents.",
            answer="SECRET WORKFLOW EXECUTION INSTRUCTIONS",
            knowledge_type=2,
            similarity=0.7,
        )

        selected = await choose_knowledge_candidate(
            "Download patent publication documents",
            [(workflow, None)],
            complete_json,
        )

        self.assertIs(selected[0], workflow)
        self.assertIn("patent publication documents", calls[0])
        self.assertNotIn("SECRET WORKFLOW EXECUTION INSTRUCTIONS", calls[0])

    async def test_low_confidence_returns_no_candidate(self):
        async def complete_json(system_prompt, user_content):
            return {
                "knowledge_id": 3,
                "confidence": 0.2,
                "reason": "weak match",
            }

        selected = await choose_knowledge_candidate(
            "unrelated request",
            [(_knowledge(3, "Patent lookup", description="Use for patents."), _tool())],
            complete_json,
        )

        self.assertIsNone(selected)

    async def test_llm_error_falls_back_to_vector_top_candidate(self):
        async def complete_json(system_prompt, user_content):
            raise RuntimeError("llm unavailable")

        top_candidate = _knowledge(4, "Top vector match", similarity=0.9)
        selected = await choose_knowledge_candidate(
            "request",
            [(top_candidate, _tool()), (_knowledge(5, "Other", similarity=0.5), _tool())],
            complete_json,
        )

        self.assertIs(selected[0], top_candidate)


if __name__ == "__main__":
    unittest.main()
