#!/usr/bin/env python3
"""Tests for session API routes (Task 6)."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_db():
    """Mock get_db_connection to return a MagicMock cursor.

    Ensures the context manager (with conn.cursor() as cur) returns
    the same cursor instance so tests can control fetchone/fetchall.
    """
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    cursor.__enter__.return_value = cursor
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    cursor.lastrowid = 1

    with patch('api_routes.session.get_db_connection', return_value=conn):
        yield conn, cursor


# ── 鉴权替身 ──────────────────────────────────────────────────
# session.py 里 verify_firebase_token 是模块级 import，patch 目标必须是
# api_routes.session.verify_firebase_token。替身行为对齐真实实现
# （sources/user/passport.py:36-37）：无 Bearer 头抛 401。
# token → uid 映射：让测试能表达"第二个用户"，从而验证"归属校验用的是
# 当前请求者的 uid"而不是某个写死的值。
TOKEN_UIDS = {
    'test-token': 123,
    'other-user-token': 456,
}
TEST_UID = 123
AUTH = {'Authorization': 'Bearer test-token'}
OTHER_AUTH = {'Authorization': 'Bearer other-user-token'}


def _fake_verify(auth_header):
    if not auth_header or not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing token')
    token = auth_header[len('Bearer '):]
    uid = TOKEN_UIDS.get(token)
    if uid is None:
        raise HTTPException(status_code=401, detail='Invalid token')
    return {'uid': uid}


@pytest.fixture
def auth():
    """把 verify_firebase_token 换成仿真替身，让测试同时覆盖 401 与已鉴权路径。"""
    with patch('api_routes.session.verify_firebase_token',
               side_effect=_fake_verify):
        yield TEST_UID


@pytest.fixture
def client(mock_db, auth):
    from fastapi import FastAPI
    from api_routes.session import register_session_routes
    import logging

    app = FastAPI()
    logger = logging.getLogger("test")
    config = MagicMock()
    router = register_session_routes(logger, config)
    app.include_router(router)
    return TestClient(app)


def test_get_session_not_found(client, mock_db):
    """GET /session/nonexistent returns 404."""
    _, cursor = mock_db
    cursor.fetchone.return_value = None

    response = client.get("/session/nonexistent", headers=AUTH)
    assert response.status_code == 404


def test_create_session(client, mock_db):
    """POST to create a session returns session_id."""
    _, cursor = mock_db
    cursor.fetchone.return_value = {
        'id': 1, 'session_id': 'sess_001', 'user_id': 123,
        'messages': '[]', 'long_task_ids': None, 'status': 1,
    }

    response = client.post("/session", json={
        "user_id": 123,
        "messages": [{"role": "user", "content": "hello"}],
    }, headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert 'session_id' in data


def test_get_user_sessions(client, mock_db):
    """GET /sessions?user_id=123 returns list."""
    import datetime
    _, cursor = mock_db
    cursor.fetchall.return_value = [
        {'session_id': 'sess_001', 'title': 'Patent Analysis',
         'status': 1, 'create_time': datetime.datetime(2026, 6, 23, 10, 0, 0),
         'update_time': datetime.datetime(2026, 6, 23, 10, 0, 0)},
    ]

    response = client.get("/sessions?user_id=123", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True
    assert len(data['sessions']) == 1


def test_append_message(client, mock_db):
    """POST /session/{id}/message appends to messages JSON."""
    _, cursor = mock_db
    cursor.fetchone.return_value = {
        'id': 1, 'session_id': 'sess_001', 'user_id': 123,
        'messages': '[{"role":"user","content":"hello"}]',
    }

    response = client.post("/session/sess_001/message", json={
        "role": "assistant",
        "content": "hi there",
    }, headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True


def test_append_message_session_not_found(client, mock_db):
    """POST /session/{id}/message on nonexistent returns 404."""
    _, cursor = mock_db
    cursor.fetchone.return_value = None

    response = client.post("/session/nonexistent/message", json={
        "role": "assistant",
        "content": "hi",
    }, headers=AUTH)
    assert response.status_code == 404


def test_archive_session(client, mock_db):
    """DELETE /session/{id} archives (status=2)."""
    _, cursor = mock_db
    cursor.rowcount = 1

    response = client.delete("/session/sess_001", headers=AUTH)
    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True


def test_archive_session_not_found(client, mock_db):
    """DELETE /session/{id} on nonexistent returns 404."""
    _, cursor = mock_db
    cursor.rowcount = 0

    response = client.delete("/session/nonexistent", headers=AUTH)
    assert response.status_code == 404


# ── 归属校验（IDOR 回归）─────────────────────────────────────

def _executed_sql(cursor):
    """把 cursor.execute 收到的 SQL 文本拼起来，供"归属过滤真的写进 SQL 了吗"断言用。"""
    return ' '.join(
        str(call.args[0]) for call in cursor.execute.call_args_list if call.args
    )


def _executed_params(cursor):
    """cursor.execute 收到的参数元组/列表，供"查的是谁"断言用。"""
    return [
        call.args[1] for call in cursor.execute.call_args_list
        if len(call.args) > 1 and isinstance(call.args[1], (tuple, list))
    ]


# 5 个本任务补鉴权的端点（方法, 路径, 合法 body）
# body 必须合法，否则 FastAPI 会在进 handler 前返回 422，测不到 401。
PROTECTED = [
    ('get', '/session/sess_001', None),
    ('get', '/session-by-id?session_id=sess_001', None),
    ('post', '/session/sess_001/message', {'role': 'user', 'content': 'x'}),
    ('put', '/session/sess_001/messages', {'messages': [], 'title': ''}),
    ('delete', '/session/sess_001', None),
]


@pytest.mark.parametrize('method,path,body', PROTECTED)
def test_endpoints_require_auth(client, method, path, body):
    """缺 Authorization 一律 401。"""
    kwargs = {'json': body} if body is not None else {}
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 401


@pytest.mark.parametrize('method,path,body', PROTECTED)
def test_endpoints_filter_by_owner(client, mock_db, method, path, body):
    """SQL 必须真的带 user_id 过滤；查不中 → 404。

    这条断言存在的理由：PUT /messages 之前注释写着 "belongs to user" 但 SQL 里
    没有 user_id。只测状态码不够，必须测 SQL 文本。
    """
    _, cursor = mock_db
    cursor.fetchone.return_value = None
    cursor.rowcount = 0
    kwargs = {'json': body} if body is not None else {}
    response = getattr(client, method)(path, headers=AUTH, **kwargs)
    assert response.status_code == 404
    assert 'user_id' in _executed_sql(cursor)
    assert any(TEST_UID in p for p in _executed_params(cursor))


def test_ownership_uses_authenticated_uid(client, mock_db):
    """用第二个用户的 token 请求时，SQL 参数里必须是他的 uid(456)，不能是别人的(123)。"""
    _, cursor = mock_db
    cursor.fetchone.return_value = None
    cursor.rowcount = 0

    response = client.get('/session/sess_001', headers=OTHER_AUTH)

    assert response.status_code == 404
    params = _executed_params(cursor)
    assert any(456 in p for p in params)
    assert not any(123 in p for p in params)


def test_save_messages_keeps_title_behavior(client, mock_db):
    """回归：PUT /messages 带 title 时仍写 title 列（web 端在用，行为不能变）。"""
    _, cursor = mock_db
    cursor.fetchone.return_value = {'id': 1}

    response = client.put('/session/sess_001/messages',
                          json={'messages': [{'role': 'user', 'content': 'hi'}],
                                'title': '新标题'},
                          headers=AUTH)

    assert response.status_code == 200
    sql = _executed_sql(cursor).lower()
    assert 'messages = %s' in sql
    assert 'title = %s' in sql


def test_save_messages_omits_title_when_blank(client, mock_db):
    """回归：title 为空串时不动 title 列（既有语义）。"""
    _, cursor = mock_db
    cursor.fetchone.return_value = {'id': 1}

    response = client.put('/session/sess_001/messages',
                          json={'messages': [], 'title': ''},
                          headers=AUTH)

    assert response.status_code == 200
    assert 'title = %s' not in _executed_sql(cursor).lower()


# ── 专用改名端点 PUT /session/{id}/title ─────────────────────

def test_rename_session(client, mock_db):
    """正常路径：改标题成功，且 SQL 带 user_id 归属过滤。"""
    _, cursor = mock_db
    cursor.rowcount = 1

    response = client.put('/session/sess_001/title',
                          json={'title': '折叠桌专利检索'},
                          headers=AUTH)

    assert response.status_code == 200
    assert response.json()['success'] is True
    sql = _executed_sql(cursor)
    assert 'user_id' in sql
    assert 'title' in sql


def test_rename_session_not_owner(client, mock_db):
    """非本人会话 → 404（不暴露"存在但不属于你"）。"""
    _, cursor = mock_db
    cursor.rowcount = 0

    response = client.put('/session/sess_001/title',
                          json={'title': 'x'}, headers=AUTH)

    assert response.status_code == 404


@pytest.mark.parametrize('bad_title', ['', '   '])
def test_rename_session_rejects_blank_title(client, mock_db, bad_title):
    """空标题在 SQL 之前就被拒，不进数据库。"""
    _, cursor = mock_db
    cursor.rowcount = 1

    response = client.put('/session/sess_001/title',
                          json={'title': bad_title}, headers=AUTH)

    assert response.status_code == 400
    assert cursor.execute.call_count == 0


def test_rename_session_requires_auth(client):
    """缺 Authorization → 401。"""
    response = client.put('/session/sess_001/title', json={'title': 'x'})
    assert response.status_code == 401


def test_rename_session_does_not_touch_messages(client, mock_db):
    """专用端点只写 title 列，绝不触碰 messages —— 这是它存在的理由。"""
    _, cursor = mock_db
    cursor.rowcount = 1

    client.put('/session/sess_001/title', json={'title': 'abc'}, headers=AUTH)

    sql = _executed_sql(cursor).lower()
    assert 'update conversations set title' in sql
    assert 'messages' not in sql
