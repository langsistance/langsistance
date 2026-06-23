import json
import pytest
from unittest.mock import patch, MagicMock
from sources.long_task.status_manager import (
    update_task_status, get_task_status, save_checkpoint,
    load_checkpoint, set_task_completed, set_task_failed,
    TASK_STATUS_PREFIX, TASK_CHECKPOINT_PREFIX, TASK_STATUS_TTL,
)


@pytest.fixture
def mock_redis():
    """Return a MagicMock that mimics redis.Redis for decode_responses=True."""
    redis_mock = MagicMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = True
    return redis_mock


def test_update_and_get_status(mock_redis):
    """Status written via update should be readable via get."""
    stored = {}
    mock_redis.set = lambda k, v, ex=None: stored.update({k: v})
    mock_redis.get = lambda k: stored.get(k)

    with patch('sources.long_task.status_manager._get_redis', return_value=mock_redis):
        update_task_status("lt_001", "analyzing", 45,
                          "分析第 5/20 个专利", table_rows=[{"patent_id": "CN001"}])
        result = get_task_status("lt_001")

    assert result['status'] == 'running'
    assert result['current_phase'] == 'analyzing'
    assert result['progress'] == 45
    assert result['current_step'] == "分析第 5/20 个专利"
    assert len(result['table_rows']) == 1


def test_checkpoint_save_and_load(mock_redis):
    """Checkpoint should survive save/load round-trip."""
    stored = {}
    mock_redis.set = lambda k, v, ex=None: stored.update({k: v})
    mock_redis.get = lambda k: stored.get(k)

    checkpoint = {
        'completed': ['CN001', 'CN003'],
        'current': 'CN005',
        'pending': ['CN005', 'CN007'],
        'completed_rows': [{'patent_id': 'CN001'}, {'patent_id': 'CN003'}],
        'failed': [],
    }

    with patch('sources.long_task.status_manager._get_redis', return_value=mock_redis):
        save_checkpoint("lt_001", checkpoint)
        result = load_checkpoint("lt_001")

    assert result['completed'] == ['CN001', 'CN003']
    assert result['pending'] == ['CN005', 'CN007']


def test_load_checkpoint_nonexistent(mock_redis):
    """Nonexistent checkpoint returns None."""
    mock_redis.get.return_value = None

    with patch('sources.long_task.status_manager._get_redis', return_value=mock_redis):
        result = load_checkpoint("nonexistent")

    assert result is None


def test_set_completed(mock_redis):
    """set_task_completed writes completed status."""
    stored = {}
    mock_redis.set = lambda k, v, ex=None: stored.update({k: v})
    mock_redis.get = lambda k: stored.get(k)

    with patch('sources.long_task.status_manager._get_redis', return_value=mock_redis):
        set_task_completed("lt_001", [{"format": "pdf", "path": "/tmp/r.pdf"}])
        result = get_task_status("lt_001")

    assert result['status'] == 'completed'
    assert result['progress'] == 100


def test_set_failed(mock_redis):
    """set_task_failed writes error status."""
    stored = {}
    mock_redis.set = lambda k, v, ex=None: stored.update({k: v})
    mock_redis.get = lambda k: stored.get(k)

    with patch('sources.long_task.status_manager._get_redis', return_value=mock_redis):
        set_task_failed("lt_001", "DI platform auth expired")
        result = get_task_status("lt_001")

    assert result['status'] == 'failed'
    assert 'DI platform auth expired' in result['error_message']
