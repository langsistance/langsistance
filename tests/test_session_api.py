#!/usr/bin/env python3
"""Tests for session API routes (Task 6)."""

import pytest
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


@pytest.fixture
def client(mock_db):
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

    response = client.get("/session/nonexistent")
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
    })
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

    response = client.get("/sessions?user_id=123")
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
    })
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
    })
    assert response.status_code == 404


def test_archive_session(client, mock_db):
    """DELETE /session/{id} archives (status=2)."""
    _, cursor = mock_db
    cursor.rowcount = 1

    response = client.delete("/session/sess_001")
    assert response.status_code == 200
    data = response.json()
    assert data['success'] is True


def test_archive_session_not_found(client, mock_db):
    """DELETE /session/{id} on nonexistent returns 404."""
    _, cursor = mock_db
    cursor.rowcount = 0

    response = client.delete("/session/nonexistent")
    assert response.status_code == 404
