import unittest


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.results = []

    def execute(self, sql, params=None):
        params = tuple(params or ())
        normalized = " ".join(sql.split()).lower()

        if normalized.startswith("select user_id") and "from users" in normalized:
            firebase_uid = params[0]
            row = self.connection.users_by_firebase_uid.get(firebase_uid)
            self.results = [row.copy()] if row else []
            return

        if normalized.startswith("select count") and "from users" in normalized:
            user_id = params[0]
            exists = user_id in self.connection.user_ids
            self.results = [{"cnt": 1 if exists else 0}]
            return

        if normalized.startswith("insert into users"):
            user_id, firebase_uid, email = params
            self.connection.user_ids.add(user_id)
            self.connection.users_by_firebase_uid[firebase_uid] = {
                "user_id": user_id,
                "firebase_uid": firebase_uid,
                "email": email,
            }
            self.connection.inserted_users.append((user_id, firebase_uid, email))
            return

        raise AssertionError(f"Unexpected SQL: {sql}")

    def fetchone(self):
        return self.results[0] if self.results else None


class FakeConnection:
    def __init__(self, users_by_firebase_uid=None, user_ids=None):
        self.users_by_firebase_uid = users_by_firebase_uid or {}
        self.user_ids = set(user_ids or ())
        self.inserted_users = []
        self.commits = 0
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expirations = []

    def get(self, key):
        return self.values.get(key)

    def setex(self, key, seconds, value):
        self.values[key] = value
        self.expirations.append((key, seconds, value))


class TestLocalUser(unittest.TestCase):
    def test_creates_user_record_when_firebase_uid_is_missing(self):
        from sources.user.local_user import ensure_local_user_record

        connection = FakeConnection(user_ids={111})
        redis_client = FakeRedis()
        generated = iter([111, 222])

        user_id = ensure_local_user_record(
            "firebase-uid-1",
            "person@example.com",
            db_factory=lambda: connection,
            redis_client=redis_client,
            random_bits=lambda bits: next(generated),
        )

        self.assertEqual(user_id, 222)
        self.assertEqual(connection.inserted_users, [(222, "firebase-uid-1", "person@example.com")])
        self.assertEqual(connection.commits, 1)
        self.assertTrue(connection.closed)
        self.assertEqual(redis_client.get("firebase_uid_firebase-uid-1"), 222)

    def test_reuses_existing_user_record_without_insert(self):
        from sources.user.local_user import ensure_local_user_record

        connection = FakeConnection(
            users_by_firebase_uid={
                "firebase-uid-1": {
                    "user_id": 333,
                    "firebase_uid": "firebase-uid-1",
                    "email": "person@example.com",
                }
            }
        )
        redis_client = FakeRedis()

        user_id = ensure_local_user_record(
            "firebase-uid-1",
            "person@example.com",
            db_factory=lambda: connection,
            redis_client=redis_client,
            random_bits=lambda bits: 999,
        )

        self.assertEqual(user_id, 333)
        self.assertEqual(connection.inserted_users, [])
        self.assertEqual(connection.commits, 0)
        self.assertEqual(redis_client.get("firebase_uid_firebase-uid-1"), 333)


if __name__ == "__main__":
    unittest.main()
