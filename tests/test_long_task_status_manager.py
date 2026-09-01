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


# ── 需求 4: set_task_failed 统一上报 long_task:fail 事件 ──

def test_set_task_failed_tracks_event_with_user_id(mock_redis):
    """状态含 user_id 时, set_task_failed 必须统一上报 long_task:fail。"""
    stored = {}
    mock_redis.set = lambda k, v, ex=None: stored.update({k: v})
    mock_redis.get = lambda k: stored.get(k)

    with patch('sources.long_task.status_manager._get_redis', return_value=mock_redis), \
         patch('sources.analytics.track_event') as mock_track:
        # 先写入含 user_id 的状态 (execute_* 开头会带)
        update_task_status("lt_fail_1", "analyzing", 10, "分析中",
                           user_id="u_42")
        set_task_failed("lt_fail_1", "no_patents_found")

    mock_track.assert_called_once()
    call_kwargs = mock_track.call_args
    assert call_kwargs.args[0] == "long_task:fail"
    assert call_kwargs.kwargs["user_id"] == "u_42"
    assert call_kwargs.kwargs["task_id"] == "lt_fail_1"


def test_set_task_failed_falls_back_to_mysql_lookup(mock_redis):
    """状态无 user_id 时 (旧提交), 从 MySQL 反查并上报。"""
    stored = {}
    mock_redis.set = lambda k, v, ex=None: stored.update({k: v})
    mock_redis.get = lambda k: stored.get(k)

    fake_conn = MagicMock()
    fake_conn.cursor.return_value.__enter__.return_value.fetchone.return_value = {
        "user_id": "u_99"}

    with patch('sources.long_task.status_manager._get_redis', return_value=mock_redis), \
         patch('sources.long_task.status_manager._lookup_task_user_id',
               return_value="u_99") as mock_lookup, \
         patch('sources.analytics.track_event') as mock_track:
        set_task_failed("lt_fail_2", "boom")

    mock_lookup.assert_called_once_with("lt_fail_2")
    mock_track.assert_called_once()
    assert mock_track.call_args.kwargs["user_id"] == "u_99"


def test_set_task_failed_silent_when_no_user_id(mock_redis):
    """查不到 user_id 时静默 (不抛、不 track)。"""
    stored = {}
    mock_redis.set = lambda k, v, ex=None: stored.update({k: v})
    mock_redis.get = lambda k: stored.get(k)

    with patch('sources.long_task.status_manager._get_redis', return_value=mock_redis), \
         patch('sources.long_task.status_manager._lookup_task_user_id',
               return_value=None), \
         patch('sources.analytics.track_event') as mock_track:
        set_task_failed("lt_fail_3", "boom")

    mock_track.assert_not_called()


def test_set_task_failed_track_error_never_breaks_state(mock_redis):
    """analytics 上报抛异常不影响任务状态写入。"""
    stored = {}
    mock_redis.set = lambda k, v, ex=None: stored.update({k: v})
    mock_redis.get = lambda k: stored.get(k)

    with patch('sources.long_task.status_manager._get_redis', return_value=mock_redis), \
         patch('sources.long_task.status_manager._lookup_task_user_id',
               return_value="u_1"), \
         patch('sources.analytics.track_event',
               side_effect=RuntimeError("analytics down")):
        set_task_failed("lt_fail_4", "boom")  # 不得抛异常

    assert json.loads(stored[list(stored)[0]])["status"] == "failed"
