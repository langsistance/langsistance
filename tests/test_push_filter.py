import unittest
import inspect


class TestPushFilter(unittest.TestCase):

    def test_query_request_accepts_push_filter(self):
        from sources.schemas import QueryRequest
        req = QueryRequest(query="test", push_filter=2)
        self.assertEqual(req.push_filter, 2)

    def test_query_request_push_filter_defaults_none(self):
        from sources.schemas import QueryRequest
        req = QueryRequest(query="test")
        self.assertIsNone(req.push_filter)

    def test_get_knowledge_tool_accepts_push_filter(self):
        from sources.knowledge.knowledge import get_knowledge_tool
        sig = inspect.signature(get_knowledge_tool)
        self.assertIn('push_filter', sig.parameters)
        self.assertIsNone(sig.parameters['push_filter'].default)

    def test_create_agent_accepts_push_filter(self):
        from sources.agents.general_agent import GeneralAgent
        sig = inspect.signature(GeneralAgent.create_agent)
        self.assertIn('push_filter', sig.parameters)
        self.assertIsNone(sig.parameters['push_filter'].default)

    def test_process_accepts_push_filter(self):
        from sources.agents.general_agent import GeneralAgent
        sig = inspect.signature(GeneralAgent.process)
        self.assertIn('push_filter', sig.parameters)
        self.assertIsNone(sig.parameters['push_filter'].default)


if __name__ == '__main__':
    unittest.main()
