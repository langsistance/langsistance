import importlib
import sys
import types
import unittest


class TestPassportUserLookup(unittest.TestCase):
    def test_get_user_by_id_returns_user_without_unbound_conn_error(self):
        sys.modules.pop("sources.user.passport", None)
        originals = {
            name: sys.modules.get(name)
            for name in ("firebase_admin", "firebase_admin.auth", "firebase_admin.credentials", "fastapi", "sources.knowledge.knowledge")
        }

        fake_firebase_admin = types.ModuleType("firebase_admin")
        fake_auth = types.ModuleType("firebase_admin.auth")
        fake_credentials = types.ModuleType("firebase_admin.credentials")
        fake_credentials.Certificate = lambda path: object()
        fake_firebase_admin.auth = fake_auth
        fake_firebase_admin.credentials = fake_credentials
        fake_firebase_admin.initialize_app = lambda cred: None

        fake_fastapi = types.ModuleType("fastapi")
        fake_fastapi.HTTPException = Exception

        class FakeCursor:
            def execute(self, sql, params):
                self.sql = sql
                self.params = params

            def fetchone(self):
                return {
                    "user_id": 9270455786494008341,
                    "firebase_uid": "firebase-uid-1",
                    "email": "person@example.com",
                    "oauth_provider": None,
                    "oauth_provider_id": None,
                    "is_active": 1,
                    "create_time": None,
                    "update_time": None,
                }

        class FakeConnection:
            def __init__(self):
                self.closed = False

            def cursor(self):
                return FakeCursor()

            def close(self):
                self.closed = True

        fake_knowledge = types.ModuleType("sources.knowledge.knowledge")
        fake_knowledge.get_redis_connection = lambda: None
        fake_knowledge.get_db_connection = lambda: FakeConnection()

        sys.modules["firebase_admin"] = fake_firebase_admin
        sys.modules["firebase_admin.auth"] = fake_auth
        sys.modules["firebase_admin.credentials"] = fake_credentials
        sys.modules["fastapi"] = fake_fastapi
        sys.modules["sources.knowledge.knowledge"] = fake_knowledge

        try:
            passport = importlib.import_module("sources.user.passport")
            user = passport.get_user_by_id("9270455786494008341")
        finally:
            sys.modules.pop("sources.user.passport", None)
            for name, module in originals.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        self.assertEqual(user["email"], "person@example.com")


if __name__ == "__main__":
    unittest.main()
