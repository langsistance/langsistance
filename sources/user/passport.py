import firebase_admin
from firebase_admin import auth, credentials
from fastapi import HTTPException
from sources.knowledge.knowledge import get_redis_connection
from datetime import datetime, timedelta, timezone
from sources.logger import Logger
import traceback
from sources.user.local_user import ensure_local_user_record

logger = Logger("passport.log")

cred = credentials.Certificate("firebase_service_key.json")
firebase_admin.initialize_app(cred)

# 初始化 Redis 连接
redis_client = get_redis_connection()  # 根据实际情况调整配置

# 白名单配置 - 字典形式
WHITELIST_TOKENS = {
    "Bearer whitelist_token_1": {
        'uid': '12957524084372015683',
        'email': 'gray.yuehui@gmail.com'
    },
    # 可以添加更多白名单项
    # "whitelist_token_2": {
    #     'uid': 12345678901234567890,
    #     'email': 'another.user@example.com'
    # }
}

def verify_firebase_token(auth_header: str):
    # 检查是否为白名单请求
    # if auth_header in WHITELIST_TOKENS:
    #     return WHITELIST_TOKENS[auth_header]

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    id_token = auth_header.split("Bearer ")[1]

    try:

        decoded_token = auth.verify_id_token(id_token)
        firebase_uid = decoded_token['uid']

        email_verified = decoded_token.get('email_verified', False)
        if not email_verified:
            raise HTTPException(status_code=401, detail="Email not verified")

        decoded_token['uid'] = ensure_local_user_record(
            firebase_uid,
            decoded_token['email'],
            redis_client=redis_client,
        )

        return decoded_token
    except Exception as e:
        logger.error(f"Error id token:{id_token}: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

def seconds_until_end_of_day() -> int:
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return int((tomorrow - now).total_seconds())

MAX_DAILY_CALLS = 100

def check_and_increase_usage(user_id: int) -> bool:
    """
    返回 True：允许调用
    返回 False：超过当日限制
    """
    today = datetime.utcnow().strftime("%Y%m%d")
    key = f"api_usage_{user_id}_{today}"

    count = redis_client.incr(key)

    if count == 1:
        # 第一次使用，设置过期时间到当天结束
        redis_client.expire(key, seconds_until_end_of_day())

    if count > MAX_DAILY_CALLS:
        return False

    return True


def get_user_by_id(user_id: str):
    """
    根据user_id查询user表，返回用户数据

    Args:
        user_id (int): 用户ID

    Returns:
        dict: 用户数据，如果用户不存在则返回None
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 查询用户数据
        query_sql = "SELECT user_id, firebase_uid, email, oauth_provider, oauth_provider_id, is_active, create_time, update_time FROM users WHERE user_id = %s"
        params = [int(user_id)]

        cursor.execute(query_sql, params)
        result = cursor.fetchone()

        if result:
            # 将查询结果转换为字典格式
            user_data = {
                'user_id': result['user_id'],
                'firebase_uid': result['firebase_uid'],
                'email': result['email'],
                'oauth_provider': result['oauth_provider'],
                'oauth_provider_id': result['oauth_provider_id'],
                'is_active': result['is_active'],
                'create_time': result['create_time'],
                'update_time': result['update_time']
            }
            return user_data
        else:
            return None

    except Exception as e:
        logger.error(f"Error querying user by ID {user_id}: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()
