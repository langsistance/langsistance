import random
from typing import Callable, Optional


DEFAULT_CACHE_TTL_SECONDS = 86400
MAX_USER_ID_ATTEMPTS = 5


class LocalUserRecordError(RuntimeError):
    pass


def _default_db_factory():
    from sources.knowledge.knowledge import get_db_connection

    return get_db_connection()


def _cache_user_id(redis_client, firebase_uid: str, user_id, cache_ttl_seconds: int) -> None:
    if redis_client is not None:
        redis_client.setex(f"firebase_uid_{firebase_uid}", cache_ttl_seconds, user_id)


def ensure_local_user_record(
    firebase_uid: str,
    email: str,
    *,
    db_factory: Optional[Callable[[], object]] = None,
    redis_client=None,
    random_bits: Optional[Callable[[int], int]] = None,
    use_cache: bool = True,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
):
    cache_key = f"firebase_uid_{firebase_uid}"
    if use_cache and redis_client is not None:
        cached_user_id = redis_client.get(cache_key)
        if cached_user_id:
            return cached_user_id

    make_connection = db_factory or _default_db_factory
    make_random_id = random_bits or random.getrandbits
    conn = make_connection()

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, email FROM users WHERE firebase_uid = %s", (firebase_uid,))
        result = cursor.fetchone()

        if result:
            user_id = result["user_id"]
            _cache_user_id(redis_client, firebase_uid, user_id, cache_ttl_seconds)
            return user_id

        attempts = 0
        user_id = None
        while attempts < MAX_USER_ID_ATTEMPTS:
            candidate_user_id = make_random_id(64)
            cursor.execute("SELECT COUNT(*) AS cnt FROM users WHERE user_id = %s", (candidate_user_id,))
            row = cursor.fetchone()
            if row and row["cnt"] > 0:
                attempts += 1
                continue

            user_id = candidate_user_id
            break

        if user_id is None:
            raise LocalUserRecordError("Failed to generate unique user_id after 5 attempts")

        cursor.execute(
            "INSERT INTO users (user_id, firebase_uid, email) VALUES (%s, %s, %s)",
            (user_id, firebase_uid, email),
        )
        conn.commit()
        # 新用户自动订阅默认场景（专利检索 scene_id=1）
        try:
            cursor.execute(
                "INSERT IGNORE INTO user_scenes (user_id, scene_id) VALUES (%s, %s)",
                (user_id, 1),
            )
            conn.commit()
        except Exception:
            pass  # 非致命：订阅失败不阻塞注册流程
        _cache_user_id(redis_client, firebase_uid, user_id, cache_ttl_seconds)
        return user_id
    finally:
        conn.close()
