#!/usr/bin/env python3
"""Tests for long task API routes (Task 7).

Endpoints:
  GET /long_task/{task_id}/status
  GET /long_task/{task_id}/report?format=pdf|docx
"""

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
    with patch('api_routes.long_task.get_task_status') as mock_get:
        mock_get.return_value = {'task_id': 'lt_unknown', 'status': 'unknown'}
        response = client.get("/long_task/lt_unknown/status")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'unknown'


def test_get_task_status_running(client):
    """GET status for running task returns full status."""
    with patch('api_routes.long_task.get_task_status') as mock_get:
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
