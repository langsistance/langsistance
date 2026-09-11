#!/usr/bin/env python3
"""Tests for long task API routes (Task 7).

Endpoints:
  GET /long_task/{task_id}/status
  GET /long_task/{task_id}/report?format=pdf|docx
"""

# api_routes.long_task 依赖 firebase_admin (本机未装) 与 REDIS_* env
import os
import sys
from unittest.mock import MagicMock

os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
sys.modules.setdefault("firebase_admin", MagicMock())

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture
def client():
    from fastapi import FastAPI
    from api_routes.long_task import register_long_task_routes
    import logging

    app = FastAPI()
    logger = logging.getLogger("test")
    config = MagicMock()
    config.get.return_value = "local"
    router = register_long_task_routes(logger, config)
    app.include_router(router)
    return TestClient(app)


def test_get_task_status_unknown(client):
    """GET status for unknown task returns unknown status."""
    with patch('api_routes.long_task.get_task_status') as mock_get,          patch('api_routes.long_task.verify_firebase_token') as mock_auth,          patch('api_routes.long_task._task_owned_by') as mock_owned:
        mock_auth.return_value = {"uid": "1"}
        mock_owned.return_value = True
        mock_get.return_value = {'task_id': 'lt_unknown', 'status': 'unknown'}
        response = client.get("/long_task/lt_unknown/status")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'unknown'


def test_get_task_status_running(client):
    """GET status for running task returns full status."""
    with patch('api_routes.long_task.get_task_status') as mock_get,          patch('api_routes.long_task.verify_firebase_token') as mock_auth,          patch('api_routes.long_task._task_owned_by') as mock_owned:
        mock_auth.return_value = {"uid": "1"}
        mock_owned.return_value = True
        mock_get.return_value = {
            'task_id': 'lt_001', 'status': 'running',
            'current_phase': 'analyzing', 'progress': 45,
            'current_step': '分析第 5/20 个专利',
            'table_columns': ['专利号', '技术领域'],
            'table_rows': [{'patent_id': 'CN001', '技术领域': 'AI'}],
        }
        response = client.get("/long_task/lt_001/status")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'running'
        assert data['current_phase'] == 'analyzing'
        assert len(data['table_rows']) == 1


def test_get_report_not_found(client):
    """GET report for unknown task returns 404."""
    with patch('api_routes.long_task.verify_firebase_token') as mock_auth, \
         patch('api_routes.long_task._task_owned_by') as mock_owned, \
         patch('api_routes.long_task.create_storage') as mock_create:
        mock_auth.return_value = {"uid": "1"}
        mock_owned.return_value = True
        mock_storage = MagicMock()
        mock_storage.get = AsyncMock(side_effect=FileNotFoundError("no file"))
        mock_create.return_value = mock_storage

        response = client.get("/long_task/lt_nonexistent/report?format=pdf")
        assert response.status_code == 404


def test_get_report_success(client):
    """GET report for completed task returns file."""
    with patch('api_routes.long_task.verify_firebase_token') as mock_auth, \
         patch('api_routes.long_task._task_owned_by') as mock_owned, \
         patch('api_routes.long_task.create_storage') as mock_create:
        mock_auth.return_value = {"uid": "1"}
        mock_owned.return_value = True
        mock_storage = MagicMock()
        mock_storage.get = AsyncMock(return_value=b"fake pdf content")
        mock_create.return_value = mock_storage

        response = client.get("/long_task/lt_001/report?format=pdf")
        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/pdf'


# ── 需求 4: POST /long_task/{task_id}/retry 一键重试 ──

def test_retry_unknown_task_returns_404(client):
    """重试不存在的任务返回 404。"""
    with patch('api_routes.long_task.verify_firebase_token') as mock_auth, \
         patch('sources.knowledge.knowledge.get_db_connection') as mock_db:
        mock_auth.return_value = {"uid": "12345"}
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value.fetchone.return_value = None
        mock_db.return_value = mock_conn

        response = client.post("/long_task/lt_nonexistent/retry")
        assert response.status_code == 404


def test_retry_creates_new_task_and_dispatches(client):
    """重试: 读原任务参数 → 新 task_id 入队 → 分发对应执行函数。"""
    input_params = '{"query": "分析专利 11701773 的审查历史", "patent_id": "11701773", "patent_source": "uspto", "lang": "zh"}'

    with patch('api_routes.long_task.verify_firebase_token') as mock_auth, \
         patch('sources.knowledge.knowledge.get_db_connection') as mock_db, \
         patch('sources.long_task.user_queue.try_start_user_task') as mock_queue, \
         patch('api_routes.long_task._dispatch_retry_task') as mock_dispatch:
        mock_auth.return_value = {"uid": "12345"}
        mock_conn = MagicMock()

        def _fetchone():
            return {
                "session_id": "sess_old",
                "scene_id": None,
                "task_type": "prosecution_analysis",
                "input_params": input_params,
            }
        mock_conn.cursor.return_value.__enter__.return_value.fetchone.side_effect = [
            _fetchone(), None,  # 第一次查原任务, 之后是 INSERT 无返回
        ]
        mock_db.return_value = mock_conn
        mock_queue.return_value = "running"

        response = client.post("/long_task/lt_old/retry")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["task_id"].startswith("lt_")
    assert data["task_id"] != "lt_old"
    assert data["status"] == "running"
    # 分发到 prosecution 执行器, 参数带原 patent_id
    mock_dispatch.assert_called_once()
    call_args = mock_dispatch.call_args
    assert call_args.args[0] == "prosecution_analysis"
    assert call_args.args[2]["patent_id"] == "11701773"
    assert call_args.args[2]["scenario"] == "prosecution"
    # 新任务行已写入 MySQL
    insert_sql = mock_conn.cursor.return_value.__enter__.return_value.execute.call_args_list[-1][0][0]
    assert "INSERT INTO long_tasks" in insert_sql


def test_retry_patent_analysis_uses_search_scenario(client):
    """patent_analysis 类型重试时 scenario 映射为 search。"""
    with patch('api_routes.long_task.verify_firebase_token') as mock_auth, \
         patch('sources.knowledge.knowledge.get_db_connection') as mock_db, \
         patch('sources.long_task.user_queue.try_start_user_task') as mock_queue, \
         patch('api_routes.long_task._dispatch_retry_task') as mock_dispatch:
        mock_auth.return_value = {"uid": "12345"}
        mock_conn = MagicMock()

        def _fetchone():
            return {
                "session_id": "sess_old",
                "scene_id": None,
                "task_type": "patent_analysis",
                "input_params": '{"query": "帮我找专利", "patent_source": "auto"}',
            }
        mock_conn.cursor.return_value.__enter__.return_value.fetchone.side_effect = [
            _fetchone(), None,
        ]
        mock_db.return_value = mock_conn
        mock_queue.return_value = "running"

        response = client.post("/long_task/lt_old2/retry")

    assert response.status_code == 200
    call_args = mock_dispatch.call_args
    assert call_args.args[0] == "patent_analysis"
    assert call_args.args[2]["scenario"] == "search"


# ── 越权修复: report 端点鉴权与归属校验 ──

def test_get_report_requires_auth(client):
    """无 Authorization 头 → 401（修复前该端点完全无鉴权）。"""
    response = client.get("/long_task/lt_001/report?format=pdf")
    assert response.status_code == 401


def test_get_report_other_users_task_returns_404(client):
    """他人任务 → 404（不是 403，不暴露"存在但不属于你"）。"""
    with patch('api_routes.long_task.verify_firebase_token') as mock_auth, \
         patch('api_routes.long_task._task_owned_by') as mock_owned:
        mock_auth.return_value = {"uid": "12345"}
        mock_owned.return_value = False
        response = client.get("/long_task/lt_other/report?format=pdf")
        assert response.status_code == 404


def test_get_report_ownership_check_uses_token_uid(client):
    """归属查询必须用 token 里的 uid，且带上 task_id —— 断言实际绑定参数，
    不只断言状态码（否则参数传错也能过）。"""
    with patch('api_routes.long_task.verify_firebase_token') as mock_auth, \
         patch('api_routes.long_task._task_owned_by') as mock_owned:
        mock_auth.return_value = {"uid": "12345"}
        mock_owned.return_value = True
        with patch('api_routes.long_task.create_storage') as mock_create:
            mock_storage = MagicMock()
            mock_storage.get = AsyncMock(return_value=b"fake pdf")
            mock_create.return_value = mock_storage
            response = client.get("/long_task/lt_mine/report?format=pdf")

    assert response.status_code == 200
    mock_owned.assert_called_once_with("lt_mine", 12345)


# ── 越权修复: status / batch_status 端点归属校验 ──

def test_status_other_users_task_returns_404(client):
    """他人任务的状态查询 → 404。"""
    with patch('api_routes.long_task.verify_firebase_token') as mock_auth, \
         patch('api_routes.long_task._task_owned_by') as mock_owned:
        mock_auth.return_value = {"uid": "12345"}
        mock_owned.return_value = False
        response = client.get("/long_task/lt_other/status")
        assert response.status_code == 404


def test_batch_status_filters_to_owned_only(client):
    """批量查询只返回本人的任务，不报错（避免被用来枚举 task_id 存在性）。"""
    with patch('api_routes.long_task.verify_firebase_token') as mock_auth, \
         patch('api_routes.long_task._task_owned_by') as mock_owned, \
         patch('api_routes.long_task.get_task_status') as mock_get:
        mock_auth.return_value = {"uid": "12345"}
        mock_owned.side_effect = lambda tid, uid: tid == "lt_mine"
        mock_get.return_value = {'task_id': 'lt_mine', 'status': 'running'}

        response = client.post("/long_task/batch_status",
                               json={"task_ids": ["lt_mine", "lt_other"]})

    assert response.status_code == 200
    statuses = response.json()["statuses"]
    assert "lt_mine" in statuses
    assert "lt_other" not in statuses
    # 只对本人任务查了状态
    mock_get.assert_called_once_with("lt_mine")


def test_batch_status_rejects_non_list_task_ids(client):
    """task_ids 不是数组时安全降级为空结果，不抛异常。"""
    with patch('api_routes.long_task.verify_firebase_token') as mock_auth:
        mock_auth.return_value = {"uid": "12345"}
        response = client.post("/long_task/batch_status", json={"task_ids": "lt_x"})
    assert response.status_code == 200
    assert response.json()["statuses"] == {}


# ── 越权修复: _task_owned_by helper 本体（其余测试都把它整体 mock 了） ──

def test_task_owned_by_binds_both_columns(client):
    """归属查询必须同时约束 task_id 与 user_id —— 只约束其一即等于无归属校验。
    直接测 helper 本体（其余测试都把它整体 mock 了，SQL 从不执行）。"""
    with patch('sources.knowledge.knowledge.get_db_connection') as mock_db:
        mock_conn = MagicMock()
        cur = mock_conn.cursor.return_value.__enter__.return_value
        cur.fetchone.return_value = {"1": 1}
        mock_db.return_value = mock_conn

        from api_routes.long_task import _task_owned_by
        result = _task_owned_by("lt_mine", 12345)

    assert result is True
    sql, params = cur.execute.call_args.args
    assert "WHERE task_id = %s AND user_id = %s" in " ".join(sql.split())
    assert params == ("lt_mine", 12345)
    mock_conn.close.assert_called_once()


def test_task_owned_by_returns_false_when_no_row(client):
    """查不到行 → False（调用方据此给 404）。"""
    with patch('sources.knowledge.knowledge.get_db_connection') as mock_db:
        mock_conn = MagicMock()
        cur = mock_conn.cursor.return_value.__enter__.return_value
        cur.fetchone.return_value = None
        mock_db.return_value = mock_conn

        from api_routes.long_task import _task_owned_by
        assert _task_owned_by("lt_other", 12345) is False
