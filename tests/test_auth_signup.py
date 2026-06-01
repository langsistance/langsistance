import importlib
import sys
import types
import unittest


class TestAuthSignup(unittest.IsolatedAsyncioTestCase):
    async def test_signup_creates_local_user_record_before_returning_token(self):
        sys.modules.pop("api_routes.auth", None)
        original_firebase_admin = sys.modules.get("firebase_admin")
        original_passport = sys.modules.get("sources.user.passport")
        original_httpx = sys.modules.get("httpx")
        original_fastapi = sys.modules.get("fastapi")
        original_pydantic = sys.modules.get("pydantic")

        fake_firebase_admin = types.ModuleType("firebase_admin")
        fake_firebase_admin.auth = types.SimpleNamespace(update_user=lambda uid, email_verified: None)
        fake_httpx = types.ModuleType("httpx")
        fake_httpx.Timeout = lambda *args, **kwargs: None
        fake_httpx.HTTPError = Exception
        fake_httpx.AsyncClient = object
        fake_fastapi = types.ModuleType("fastapi")

        class FakeHTTPException(Exception):
            def __init__(self, status_code=None, detail=None):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class FakeAPIRouter:
            def post(self, *args, **kwargs):
                return lambda func: func

        fake_fastapi.APIRouter = FakeAPIRouter
        fake_fastapi.HTTPException = FakeHTTPException
        fake_pydantic = types.ModuleType("pydantic")

        class FakeBaseModel:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        fake_pydantic.BaseModel = FakeBaseModel
        fake_pydantic.EmailStr = str
        sys.modules["firebase_admin"] = fake_firebase_admin
        sys.modules["httpx"] = fake_httpx
        sys.modules["fastapi"] = fake_fastapi
        sys.modules["pydantic"] = fake_pydantic
        sys.modules["sources.user.passport"] = types.ModuleType("sources.user.passport")

        try:
            auth_route = importlib.import_module("api_routes.auth")
        finally:
            if original_firebase_admin is not None:
                sys.modules["firebase_admin"] = original_firebase_admin
            else:
                sys.modules.pop("firebase_admin", None)

            if original_passport is not None:
                sys.modules["sources.user.passport"] = original_passport
            else:
                sys.modules.pop("sources.user.passport", None)

            if original_httpx is not None:
                sys.modules["httpx"] = original_httpx
            else:
                sys.modules.pop("httpx", None)

            if original_fastapi is not None:
                sys.modules["fastapi"] = original_fastapi
            else:
                sys.modules.pop("fastapi", None)

            if original_pydantic is not None:
                sys.modules["pydantic"] = original_pydantic
            else:
                sys.modules.pop("pydantic", None)

        firebase_calls = []
        local_user_calls = []

        async def fake_firebase_post(url, payload):
            firebase_calls.append((url, payload))
            if url.endswith(":signUp"):
                return {"localId": "firebase-uid-1"}
            return {
                "idToken": "fresh-id-token",
                "refreshToken": "refresh-token",
                "expiresIn": "3600",
                "localId": "firebase-uid-1",
                "email": "person@example.com",
            }

        def fake_ensure_local_user_record(firebase_uid, email, **kwargs):
            local_user_calls.append((firebase_uid, email, kwargs))
            return 12345

        auth_route._firebase_post = fake_firebase_post
        auth_route.fb_admin_auth = types.SimpleNamespace(update_user=lambda uid, email_verified: None)
        auth_route.ensure_local_user_record = fake_ensure_local_user_record

        response = await auth_route.auth_signup(
            auth_route.EmailPasswordRequest(email="person@example.com", password="secret123")
        )

        self.assertEqual(len(local_user_calls), 1)
        self.assertEqual(local_user_calls[0][0:2], ("firebase-uid-1", "person@example.com"))
        self.assertIsNone(local_user_calls[0][2]["redis_client"])
        self.assertIs(local_user_calls[0][2]["use_cache"], False)
        self.assertEqual(response["idToken"], "fresh-id-token")
        self.assertEqual(response["localId"], "firebase-uid-1")
        self.assertEqual(len(firebase_calls), 2)


if __name__ == "__main__":
    unittest.main()
