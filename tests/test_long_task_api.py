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
    with patch('api_routes.long_task.get_task_status') as mock_get,          patch('api_routes.long_task.verify_firebase_token') as mock_auth:
        mock_auth.return_value = {"uid": "1"}
        mock_get.return_value = {'task_id': 'lt_unknown', 'status': 'unknown'}
        response = client.get("/long_task/lt_unknown/status")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'unknown'


def test_get_task_status_running(client):
    """GET status for running task returns full status."""
    with patch('api_routes.long_task.get_task_status') as mock_get,          patch('api_routes.long_task.verify_firebase_token') as mock_auth:
        mock_auth.return_value = {"uid": "1"}
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
    with patch('api_routes.long_task.create_storage') as mock_create:
        mock_storage = MagicMock()
        mock_storage.get = AsyncMock(side_effect=FileNotFoundError("no file"))
        mock_create.return_value = mock_storage

        response = client.get("/long_task/lt_nonexistent/report?format=pdf")
        assert response.status_code == 404


def test_get_report_success(client):
    """GET report for completed task returns file."""
    with patch('api_routes.long_task.create_storage') as mock_create:
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
