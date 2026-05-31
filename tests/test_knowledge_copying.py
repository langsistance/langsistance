import json
import unittest


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self._results = []
        self.lastrowid = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        params = tuple(params or ())
        normalized = " ".join(sql.split()).lower()

        if normalized.startswith("select") and "from knowledge" in normalized:
            knowledge_id = params[0]
            row = self.connection.knowledge.get(knowledge_id)
            self._results = [row.copy()] if row and row.get("status") == 1 else []
            return

        if normalized.startswith("select") and "from tools" in normalized:
            tool_id = params[0]
            row = self.connection.tools.get(tool_id)
            self._results = [row.copy()] if row and row.get("status") == 1 else []
            return

        if normalized.startswith("insert into tools"):
            self.connection.next_tool_id += 1
            tool_id = self.connection.next_tool_id
            self.connection.tools[tool_id] = {
                "id": tool_id,
                "user_id": params[0],
                "title": params[1],
                "description": params[2],
                "url": params[3],
                "push": params[4],
                "status": params[5],
                "timeout": params[6],
                "params": params[7],
            }
            self.connection.inserted_tools.append(self.connection.tools[tool_id])
            self.lastrowid = tool_id
            return

        if normalized.startswith("insert into knowledge"):
            self.connection.next_knowledge_id += 1
            knowledge_id = self.connection.next_knowledge_id
            self.connection.knowledge[knowledge_id] = {
                "id": knowledge_id,
                "user_id": params[0],
                "question": params[1],
                "description": params[2],
                "answer": params[3],
                "public": params[4],
                "status": params[5],
                "embedding_id": params[6],
                "model_name": params[7],
                "tool_id": params[8],
                "params": params[9],
                "type": params[10],
            }
            self.connection.inserted_knowledge.append(self.connection.knowledge[knowledge_id])
            self.lastrowid = knowledge_id
            return

        raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchone(self):
        return self._results[0] if self._results else None

    def fetchall(self):
        return list(self._results)


class FakeConnection:
    def __init__(self, knowledge=None, tools=None):
        self.knowledge = knowledge or {}
        self.tools = tools or {}
        self.inserted_knowledge = []
        self.inserted_tools = []
        self.next_knowledge_id = 1000
        self.next_tool_id = 2000
        self.begins = 0
        self.commits = 0
        self.rollbacks = 0

    def begin(self):
        self.begins += 1

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def cursor(self):
        return FakeCursor(self)


def normal_knowledge(knowledge_id, tool_id):
    return {
        "id": knowledge_id,
        "user_id": "source-user",
        "question": f"normal {knowledge_id}",
        "description": "",
        "answer": f"answer {knowledge_id}",
        "public": 2,
        "model_name": "gpt-4o-mini",
        "tool_id": tool_id,
        "params": "{}",
        "type": 1,
        "status": 1,
    }


def backend_tool(tool_id, push=2):
    return {
        "id": tool_id,
        "user_id": "source-user",
        "title": f"tool {tool_id}",
        "description": "",
        "url": "https://example.test/tool",
        "push": push,
        "public": 2,
        "status": 1,
        "timeout": 30,
        "params": "{}",
    }


class TestKnowledgeCopying(unittest.TestCase):
    def test_copies_workflow_dependencies_reusing_push2_tools_and_remapping_steps(self):
        from sources.knowledge.copying import copy_knowledge_to_user

        workflow_params = {
            "type": "workflow",
            "version": 1,
            "mode": "context_chain",
            "steps": [
                {"id": "step_1", "knowledge_id": 101},
                {"id": "step_2", "knowledge_id": 102},
            ],
        }
        connection = FakeConnection(
            knowledge={
                10: {
                    "id": 10,
                    "user_id": "source-user",
                    "question": "workflow",
                    "description": "",
                    "answer": "workflow answer",
                    "public": 2,
                    "model_name": "gpt-4o-mini",
                    "tool_id": 0,
                    "params": json.dumps(workflow_params),
                    "type": 2,
                    "status": 1,
                },
                101: normal_knowledge(101, 201),
                102: normal_knowledge(102, 202),
            },
            tools={
                201: backend_tool(201, push=2),
                202: backend_tool(202, push=2),
            },
        )
        embedded = []

        new_id = copy_knowledge_to_user(
            connection,
            source_knowledge_id=10,
            target_user_id="target-user",
            embedding_writer=lambda knowledge_id, question, description, answer: embedded.append(
                (knowledge_id, question, description, answer)
            ),
        )

        self.assertEqual(new_id, 1003)
        self.assertEqual(connection.begins, 1)
        self.assertEqual(connection.commits, 1)
        self.assertEqual(connection.rollbacks, 0)
        self.assertEqual(connection.inserted_tools, [])
        copied_first, copied_second, copied_workflow = connection.inserted_knowledge
        self.assertEqual(copied_first["tool_id"], 201)
        self.assertEqual(copied_second["tool_id"], 202)
        self.assertEqual(copied_workflow["tool_id"], 0)
        self.assertEqual(copied_workflow["type"], 2)
        self.assertEqual(copied_workflow["public"], 1)
        remapped_params = json.loads(copied_workflow["params"])
        self.assertEqual(remapped_params["steps"][0]["knowledge_id"], copied_first["id"])
        self.assertEqual(remapped_params["steps"][1]["knowledge_id"], copied_second["id"])
        self.assertEqual(
            embedded,
            [
                (copied_first["id"], copied_first["question"], copied_first["description"], copied_first["answer"]),
                (copied_second["id"], copied_second["question"], copied_second["description"], copied_second["answer"]),
                (copied_workflow["id"], copied_workflow["question"], copied_workflow["description"], copied_workflow["answer"]),
            ],
        )

    def test_copies_duplicate_workflow_dependency_once(self):
        from sources.knowledge.copying import copy_knowledge_to_user

        connection = FakeConnection(
            knowledge={
                10: {
                    "id": 10,
                    "user_id": "source-user",
                    "question": "workflow",
                    "description": "",
                    "answer": "workflow answer",
                    "public": 2,
                    "model_name": "gpt-4o-mini",
                    "tool_id": 0,
                    "params": json.dumps({
                        "type": "workflow",
                        "version": 1,
                        "mode": "context_chain",
                        "steps": [
                            {"id": "step_1", "knowledge_id": 101},
                            {"id": "step_2", "knowledge_id": 101},
                        ],
                    }),
                    "type": 2,
                    "status": 1,
                },
                101: normal_knowledge(101, 201),
            },
            tools={201: backend_tool(201, push=2)},
        )

        new_id = copy_knowledge_to_user(connection, 10, "target-user")

        self.assertEqual(new_id, 1002)
        self.assertEqual(len(connection.inserted_knowledge), 2)
        copied_dependency, copied_workflow = connection.inserted_knowledge
        remapped_params = json.loads(copied_workflow["params"])
        self.assertEqual(remapped_params["steps"][0]["knowledge_id"], copied_dependency["id"])
        self.assertEqual(remapped_params["steps"][1]["knowledge_id"], copied_dependency["id"])

    def test_rejects_workflow_dependency_without_push2_tool_and_rolls_back(self):
        from sources.knowledge.copying import KnowledgeCopyError, copy_knowledge_to_user

        connection = FakeConnection(
            knowledge={
                10: {
                    "id": 10,
                    "user_id": "source-user",
                    "question": "workflow",
                    "description": "",
                    "answer": "workflow answer",
                    "public": 2,
                    "model_name": "gpt-4o-mini",
                    "tool_id": 0,
                    "params": json.dumps({
                        "type": "workflow",
                        "version": 1,
                        "mode": "context_chain",
                        "steps": [{"id": "step_1", "knowledge_id": 101}],
                    }),
                    "type": 2,
                    "status": 1,
                },
                101: normal_knowledge(101, 201),
            },
            tools={201: backend_tool(201, push=1)},
        )

        with self.assertRaisesRegex(KnowledgeCopyError, "push=2"):
            copy_knowledge_to_user(connection, 10, "target-user")

        self.assertEqual(connection.commits, 0)
        self.assertEqual(connection.rollbacks, 1)
        self.assertEqual(connection.inserted_knowledge, [])
        self.assertEqual(connection.inserted_tools, [])

    def test_validates_all_workflow_dependencies_before_inserting_any_records(self):
        from sources.knowledge.copying import KnowledgeCopyError, copy_knowledge_to_user

        connection = FakeConnection(
            knowledge={
                10: {
                    "id": 10,
                    "user_id": "source-user",
                    "question": "workflow",
                    "description": "",
                    "answer": "workflow answer",
                    "public": 2,
                    "model_name": "gpt-4o-mini",
                    "tool_id": 0,
                    "params": json.dumps({
                        "type": "workflow",
                        "version": 1,
                        "mode": "context_chain",
                        "steps": [
                            {"id": "step_1", "knowledge_id": 101},
                            {"id": "step_2", "knowledge_id": 102},
                        ],
                    }),
                    "type": 2,
                    "status": 1,
                },
                101: normal_knowledge(101, 201),
                102: normal_knowledge(102, 202),
            },
            tools={
                201: backend_tool(201, push=2),
                202: backend_tool(202, push=1),
            },
        )
        embedded = []

        with self.assertRaisesRegex(KnowledgeCopyError, "push=2"):
            copy_knowledge_to_user(
                connection,
                10,
                "target-user",
                embedding_writer=lambda knowledge_id, question, description, answer: embedded.append(knowledge_id),
            )

        self.assertEqual(connection.commits, 0)
        self.assertEqual(connection.rollbacks, 1)
        self.assertEqual(connection.inserted_knowledge, [])
        self.assertEqual(connection.inserted_tools, [])
        self.assertEqual(embedded, [])

    def test_normal_copy_reuses_push2_tool_and_copies_non_push2_tool(self):
        from sources.knowledge.copying import copy_knowledge_to_user

        push2_connection = FakeConnection(
            knowledge={101: normal_knowledge(101, 201)},
            tools={201: backend_tool(201, push=2)},
        )

        push2_new_id = copy_knowledge_to_user(push2_connection, 101, "target-user")

        self.assertEqual(push2_new_id, 1001)
        self.assertEqual(push2_connection.inserted_tools, [])
        self.assertEqual(push2_connection.inserted_knowledge[0]["tool_id"], 201)

        owned_tool_connection = FakeConnection(
            knowledge={102: normal_knowledge(102, 202)},
            tools={202: backend_tool(202, push=1)},
        )

        owned_tool_new_id = copy_knowledge_to_user(owned_tool_connection, 102, "target-user")

        self.assertEqual(owned_tool_new_id, 1001)
        self.assertEqual(len(owned_tool_connection.inserted_tools), 1)
        self.assertEqual(owned_tool_connection.inserted_tools[0]["user_id"], "target-user")
        self.assertEqual(owned_tool_connection.inserted_knowledge[0]["tool_id"], 2001)


if __name__ == "__main__":
    unittest.main()
