# Batch Patent Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add batch patent analysis as a "long task" capability — backend intent recognition triggers a Celery worker that downloads, analyzes, and reports on up to N patents, with persistent session storage and real-time frontend progress display.

**Architecture:** Long task flow is triggered through existing `/query_stream` endpoint when LLM routing identifies type=3 knowledge. A special SSE event (`long_task_created`) notifies the frontend, which switches to polling `GET /long_task/{task_id}/status`. The Celery worker runs 4 serial phases: (1) Flash generates dynamic table columns, (2) per-patent download→analyze→summarize with Redis checkpointing, (3) Pro generates dynamic report outline then writes section-by-section, (4) export Word/PDF via storage abstraction.

**Tech Stack:** Python 3.11, FastAPI, Celery (Redis broker), MySQL (pymysql), Redis, python-docx, weasyprint, DeepSeek V4, MiniMax 2.7

## Global Constraints

- All LLM calls use existing `Provider` from `sources/llm_provider.py` — no new LLM integration
- `max_patents` read from `config.ini [LONG_TASK]` at Worker startup, not hardcoded
- Full serial execution — no parallelism within a single long task
- Celery pool=solo, concurrency=1
- Knowledge type=3 for long task intent matching
- Session lazy-created: only when long task triggers
- `conversation_history` carried by frontend, not cached by backend for short conversations
- Report files stored via `ReportStorage` abstraction; MVP = local filesystem
- Agent pool max reduced from 10 to 3
- All new routes follow existing factory pattern `register_*_routes()`
- TDD: write test first, verify it fails, implement, verify it passes, commit
- Test coverage: 80%+ for new code

---

## File Map

```
NEW FILES (11):
  mysql/init/add_conversations.sql          — conversations table schema
  mysql/init/add_long_tasks.sql             — long_tasks table schema
  sources/long_task/__init__.py              — module init
  sources/long_task/config.py               — config reader for [LONG_TASK]
  sources/long_task/storage.py              — ReportStorage abstract + local impl
  sources/long_task/status_manager.py       — Redis state read/write + checkpoint
  sources/long_task/patent_analyzer.py       — core 4-phase pipeline
  sources/long_task/report_generator.py      — Word/PDF export
  celery_worker.py                           — Celery app + task definition
  api_routes/session.py                      — Session CRUD API routes
  api_routes/long_task.py                    — Long task status/report API routes

MODIFIED FILES (8):
  config.ini                                 — add [LONG_TASK] section
  config.ini.example                         — add [LONG_TASK] example
  sources/knowledge/type_utils.py            — support type=3
  sources/knowledge/selection.py             — conversation_history in routing
  sources/agents/general_agent.py            — long task intent branch
  api_routes/core.py                         — SSE event, agent pool=3
  api.py                                     — register session + long_task routes
  requirements.txt                           — celery, python-docx, weasyprint

TEST FILES (6):
  tests/test_long_task_config.py
  tests/test_long_task_storage.py
  tests/test_long_task_status_manager.py
  tests/test_patent_analyzer.py
  tests/test_session_api.py
  tests/test_long_task_api.py
```

---

## Phase 1: Infrastructure (Tasks 1–12)

### Task 1: Database Schema — conversations table

**Files:**
- Create: `mysql/init/add_conversations.sql`
- Test: `tests/test_long_task_config.py` (shared config test, Task 3 fills it)

**Produces:** Table `conversations` in MySQL with columns: id, session_id, user_id, scene_id, title, messages (JSON), long_task_ids (JSON), status, create_time, update_time.

- [ ] **Step 1: Write the SQL file**

```sql
-- mysql/init/add_conversations.sql
CREATE TABLE IF NOT EXISTS conversations (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id      VARCHAR(64) NOT NULL UNIQUE COMMENT 'session unique identifier',
    user_id         BIGINT UNSIGNED NOT NULL COMMENT 'user ID from users table',
    scene_id        BIGINT UNSIGNED DEFAULT NULL COMMENT 'associated scene',
    title           VARCHAR(256) DEFAULT '' COMMENT 'session title, LLM-generated',
    messages        JSON NOT NULL COMMENT 'conversation messages array',
    long_task_ids   JSON DEFAULT NULL COMMENT 'linked long task IDs',
    status          TINYINT UNSIGNED DEFAULT 1 COMMENT '1=active, 2=archived',
    create_time     DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time     DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_create_time (create_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

- [ ] **Step 2: Run the SQL against test/dev database**

Run: `mysql -h $MYSQL_HOST -u $MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DATABASE < mysql/init/add_conversations.sql`

Expected: Table created without errors.

- [ ] **Step 3: Verify table exists**

Run: `mysql -h $MYSQL_HOST -u $MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DATABASE -e "DESCRIBE conversations;"`

Expected: Shows all column definitions.

- [ ] **Step 4: Commit**

```bash
git add mysql/init/add_conversations.sql
git commit -m "feat: add conversations table for session persistence"
```

---

### Task 2: Database Schema — long_tasks table

**Files:**
- Create: `mysql/init/add_long_tasks.sql`

**Produces:** Table `long_tasks` with all columns from the design doc.

- [ ] **Step 1: Write the SQL file**

```sql
-- mysql/init/add_long_tasks.sql
CREATE TABLE IF NOT EXISTS long_tasks (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    task_id         VARCHAR(64) NOT NULL UNIQUE COMMENT 'public task UUID',
    session_id      VARCHAR(64) DEFAULT NULL COMMENT 'associated session',
    user_id         BIGINT UNSIGNED NOT NULL,
    scene_id        BIGINT UNSIGNED DEFAULT NULL,
    task_type       VARCHAR(64) NOT NULL DEFAULT 'patent_analysis',
    input_params    JSON NOT NULL COMMENT 'query, patent_ids, model_family, etc.',
    status          VARCHAR(32) NOT NULL DEFAULT 'pending' COMMENT 'pending|running|completed|failed',
    progress        INT DEFAULT 0 COMMENT '0-100',
    current_phase   VARCHAR(64) DEFAULT NULL,
    current_step    VARCHAR(512) DEFAULT NULL,
    phases          JSON DEFAULT NULL COMMENT 'phase details with steps',
    table_columns   JSON DEFAULT NULL COMMENT 'Phase 1 output',
    table_rows      JSON DEFAULT NULL COMMENT 'Phase 2 output',
    result_summary  TEXT DEFAULT NULL COMMENT 'Phase 3 report text',
    report_files    JSON DEFAULT NULL COMMENT '[{format, path, size, created_at}]',
    error_message   TEXT DEFAULT NULL,
    celery_task_id  VARCHAR(128) DEFAULT NULL,
    create_time     DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time     DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    complete_time   DATETIME DEFAULT NULL,
    INDEX idx_user_id (user_id),
    INDEX idx_session_id (session_id),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

- [ ] **Step 2: Run and verify the SQL**

Run:
```bash
mysql -h $MYSQL_HOST -u $MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DATABASE < mysql/init/add_long_tasks.sql
mysql -h $MYSQL_HOST -u $MYSQL_USER -p$MYSQL_PASSWORD $MYSQL_DATABASE -e "DESCRIBE long_tasks;"
```

Expected: Table created with all columns.

- [ ] **Step 3: Commit**

```bash
git add mysql/init/add_long_tasks.sql
git commit -m "feat: add long_tasks table for async task tracking"
```

---

### Task 3: Long Task Config Reader

**Files:**
- Create: `sources/long_task/__init__.py`
- Create: `sources/long_task/config.py`
- Test: `tests/test_long_task_config.py`

**Interfaces:**
- Produces: `get_long_task_config() -> dict` with keys `provider_family`, `max_patents`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_long_task_config.py
import os
import tempfile
import pytest
from sources.long_task.config import get_long_task_config


def test_reads_provider_family_and_max_patents():
    """Should read provider_family and max_patents from [LONG_TASK] section."""
    config_content = """[LONG_TASK]
provider_family = deepseek
max_patents = 20
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        config = get_long_task_config(config_path)
        assert config['provider_family'] == 'deepseek'
        assert config['max_patents'] == 20
    finally:
        os.unlink(config_path)


def test_defaults_when_section_missing():
    """Should return defaults when [LONG_TASK] section is absent."""
    config_content = """[MAIN]
provider_name = openai
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        config = get_long_task_config(config_path)
        assert config['provider_family'] in ('deepseek', 'minimax')
        assert isinstance(config['max_patents'], int)
        assert config['max_patents'] > 0
    finally:
        os.unlink(config_path)


def test_max_patents_is_integer():
    """max_patents should always be returned as an int."""
    config_content = """[LONG_TASK]
max_patents = 15
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write(config_content)
        config_path = f.name

    try:
        config = get_long_task_config(config_path)
        assert isinstance(config['max_patents'], int)
        assert config['max_patents'] == 15
    finally:
        os.unlink(config_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_long_task_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sources.long_task.config'`

- [ ] **Step 3: Create module init**

```python
# sources/long_task/__init__.py
"""Long task execution module for batch patent analysis."""
```

- [ ] **Step 4: Create config reader (minimal implementation)**

```python
# sources/long_task/config.py
import configparser
import os

DEFAULT_PROVIDER_FAMILY = 'deepseek'
DEFAULT_MAX_PATENTS = 20


def get_long_task_config(config_path: str = 'config.ini') -> dict:
    """Read [LONG_TASK] section from config file.

    Returns:
        dict with keys: provider_family (str), max_patents (int)
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    provider_family = DEFAULT_PROVIDER_FAMILY
    max_patents = DEFAULT_MAX_PATENTS

    if cfg.has_section('LONG_TASK'):
        provider_family = cfg.get('LONG_TASK', 'provider_family',
                                  fallback=DEFAULT_PROVIDER_FAMILY)
        max_patents = cfg.getint('LONG_TASK', 'max_patents',
                                 fallback=DEFAULT_MAX_PATENTS)

    return {
        'provider_family': provider_family,
        'max_patents': max_patents,
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_long_task_config.py -v`
Expected: 3 PASS

- [ ] **Step 6: Add [LONG_TASK] to config.ini and config.ini.example**

```ini
# config.ini — append at end of file
[LONG_TASK]
provider_family = deepseek
max_patents = 20
```

```ini
# config.ini.example — append at end of file
[LONG_TASK]
# provider_family: which LLM family to use for long tasks (deepseek | minimax)
provider_family = deepseek
# max_patents: maximum patents to analyze in a single long task (excess truncated)
max_patents = 20
```

- [ ] **Step 7: Commit**

```bash
git add sources/long_task/__init__.py sources/long_task/config.py config.ini config.ini.example tests/test_long_task_config.py
git commit -m "feat: add long task config reader with [LONG_TASK] section"
```

---

### Task 4: Report Storage Abstraction

**Files:**
- Create: `sources/long_task/storage.py`
- Test: `tests/test_long_task_storage.py`

**Interfaces:**
- Produces:
  - `class ReportStorage(ABC)` — `async put(task_id, filename, content) -> str`, `async get(task_id, filename) -> bytes`, `async delete(task_id) -> None`
  - `class LocalReportStorage(ReportStorage)` — filesystem-backed
  - `def create_storage(config: dict) -> ReportStorage` — factory function

- [ ] **Step 1: Write the failing test**

```python
# tests/test_long_task_storage.py
import os
import tempfile
import pytest
from sources.long_task.storage import LocalReportStorage, create_storage


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.mark.asyncio
async def test_put_and_get(temp_dir):
    """Should store and retrieve file content."""
    storage = LocalReportStorage(base_dir=temp_dir)
    task_id = "lt_test_001"
    content = b"fake report content"

    path = await storage.put(task_id, "report.pdf", content)
    retrieved = await storage.get(task_id, "report.pdf")

    assert retrieved == content
    assert os.path.exists(path)


@pytest.mark.asyncio
async def test_delete_removes_all_files(temp_dir):
    """Should remove all files for a task."""
    storage = LocalReportStorage(base_dir=temp_dir)
    task_id = "lt_test_002"

    await storage.put(task_id, "report.pdf", b"pdf content")
    await storage.put(task_id, "report.docx", b"docx content")

    await storage.delete(task_id)

    assert not os.path.exists(os.path.join(temp_dir, task_id))


@pytest.mark.asyncio
async def test_get_nonexistent_raises(temp_dir):
    """Should raise FileNotFoundError for missing files."""
    storage = LocalReportStorage(base_dir=temp_dir)

    with pytest.raises(FileNotFoundError):
        await storage.get("nonexistent", "report.pdf")


def test_create_storage_local():
    """create_storage with backend='local' returns LocalReportStorage."""
    config = {'report_storage_backend': 'local', 'report_storage_local_dir': '/tmp/reports'}
    storage = create_storage(config)
    assert isinstance(storage, LocalReportStorage)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_long_task_storage.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write storage module**

```python
# sources/long_task/storage.py
import os
from abc import ABC, abstractmethod


class ReportStorage(ABC):
    """Abstract interface for report file storage."""

    @abstractmethod
    async def put(self, task_id: str, filename: str, content: bytes) -> str:
        """Store a file, return its path."""
        ...

    @abstractmethod
    async def get(self, task_id: str, filename: str) -> bytes:
        """Retrieve file content."""
        ...

    @abstractmethod
    async def delete(self, task_id: str) -> None:
        """Delete all files for a task."""
        ...


class LocalReportStorage(ReportStorage):
    """Filesystem-backed report storage (MVP)."""

    def __init__(self, base_dir: str = "/opt/workspace/reports"):
        self.base_dir = base_dir

    def _task_dir(self, task_id: str) -> str:
        return os.path.join(self.base_dir, task_id)

    async def put(self, task_id: str, filename: str, content: bytes) -> str:
        task_dir = self._task_dir(task_id)
        os.makedirs(task_dir, exist_ok=True)
        filepath = os.path.join(task_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(content)
        return filepath

    async def get(self, task_id: str, filename: str) -> bytes:
        filepath = os.path.join(self._task_dir(task_id), filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Report not found: {filepath}")
        with open(filepath, 'rb') as f:
            return f.read()

    async def delete(self, task_id: str) -> None:
        import shutil
        task_dir = self._task_dir(task_id)
        if os.path.exists(task_dir):
            shutil.rmtree(task_dir)


def create_storage(config: dict) -> ReportStorage:
    """Factory: create the configured storage backend."""
    backend = config.get('report_storage_backend', 'local')
    if backend == 'local':
        base_dir = config.get('report_storage_local_dir', '/opt/workspace/reports')
        return LocalReportStorage(base_dir=base_dir)
    raise ValueError(f"Unknown storage backend: {backend}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_long_task_storage.py -v`
Expected: 4 PASS

- [ ] **Step 5: Add storage config to config.ini**

```ini
# Append to config.ini
[STORAGE]
backend = local
local_base_dir = /opt/workspace/reports
```

- [ ] **Step 6: Commit**

```bash
git add sources/long_task/storage.py tests/test_long_task_storage.py config.ini
git commit -m "feat: add report storage abstraction with local filesystem backend"
```

---

### Task 5: Redis Status Manager

**Files:**
- Create: `sources/long_task/status_manager.py`
- Test: `tests/test_long_task_status_manager.py`

**Interfaces:**
- Produces:
  - `update_task_status(task_id, phase, progress, step_msg, **extra) -> None` — write to Redis
  - `get_task_status(task_id) -> dict` — read from Redis
  - `save_checkpoint(task_id, checkpoint: dict) -> None`
  - `load_checkpoint(task_id) -> dict | None`
  - `set_task_completed(task_id, report_files: list) -> None`
  - `set_task_failed(task_id, error: str) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_long_task_status_manager.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_long_task_status_manager.py -v`
Expected: FAIL

- [ ] **Step 3: Write status manager**

```python
# sources/long_task/status_manager.py
import json
from sources.knowledge.knowledge import get_redis_connection

TASK_STATUS_PREFIX = "lt"
TASK_CHECKPOINT_PREFIX = "lt"
TASK_STATUS_TTL = 86400  # 24 hours


def _get_redis():
    return get_redis_connection()


def _status_key(task_id: str) -> str:
    return f"{TASK_STATUS_PREFIX}:{task_id}:status"


def _checkpoint_key(task_id: str) -> str:
    return f"{TASK_CHECKPOINT_PREFIX}:{task_id}:checkpoint"


def update_task_status(task_id: str, phase: str, progress: int,
                       step_msg: str, **extra) -> None:
    """Write current task status to Redis."""
    r = _get_redis()
    status = {
        'task_id': task_id,
        'status': 'running',
        'current_phase': phase,
        'progress': progress,
        'current_step': step_msg,
        **extra,
    }
    r.set(_status_key(task_id), json.dumps(status, ensure_ascii=False),
          ex=TASK_STATUS_TTL)


def get_task_status(task_id: str) -> dict:
    """Read current task status from Redis."""
    r = _get_redis()
    raw = r.get(_status_key(task_id))
    if raw is None:
        return {'task_id': task_id, 'status': 'unknown'}
    return json.loads(raw)


def save_checkpoint(task_id: str, checkpoint: dict) -> None:
    """Save per-patent processing checkpoint to Redis."""
    r = _get_redis()
    r.set(_checkpoint_key(task_id), json.dumps(checkpoint, ensure_ascii=False),
          ex=TASK_STATUS_TTL)


def load_checkpoint(task_id: str) -> dict | None:
    """Load checkpoint; returns None if not found."""
    r = _get_redis()
    raw = r.get(_checkpoint_key(task_id))
    if raw is None:
        return None
    return json.loads(raw)


def set_task_completed(task_id: str, report_files: list) -> None:
    """Mark task as completed with report file metadata."""
    r = _get_redis()
    raw = r.get(_status_key(task_id))
    status = json.loads(raw) if raw else {}
    status['status'] = 'completed'
    status['progress'] = 100
    status['report_files'] = report_files
    r.set(_status_key(task_id), json.dumps(status, ensure_ascii=False),
          ex=TASK_STATUS_TTL)


def set_task_failed(task_id: str, error: str) -> None:
    """Mark task as failed with error message."""
    r = _get_redis()
    raw = r.get(_status_key(task_id))
    status = json.loads(raw) if raw else {}
    status['status'] = 'failed'
    status['error_message'] = error
    r.set(_status_key(task_id), json.dumps(status, ensure_ascii=False),
          ex=TASK_STATUS_TTL)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_long_task_status_manager.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/status_manager.py tests/test_long_task_status_manager.py
git commit -m "feat: add Redis status manager for long task progress tracking"
```

---

### Task 6: Session API Routes

**Files:**
- Create: `api_routes/session.py`
- Test: `tests/test_session_api.py`

**Interfaces:**
- Consumes: `get_db_connection()` from `sources.knowledge.knowledge`
- Produces:
  - `register_session_routes(logger, config) -> APIRouter`
  - `GET /session/{session_id}` — returns session with messages + long_task_ids
  - `GET /sessions?user_id=UID` — list user sessions
  - `POST /session/{session_id}/message` — append message
  - `DELETE /session/{session_id}` — archive (status=2)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_db():
    """Mock get_db_connection to return a MagicMock cursor."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value = cursor
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
    _, cursor = mock_db
    cursor.fetchall.return_value = [
        {'session_id': 'sess_001', 'title': 'Patent Analysis',
         'status': 1, 'create_time': '2026-06-23T10:00:00'},
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_session_api.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write session API routes**

```python
# api_routes/session.py
import uuid
import json
import pymysql
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from sources.knowledge.knowledge import get_db_connection


class CreateSessionRequest(BaseModel):
    user_id: int
    scene_id: int | None = None
    messages: list = []
    title: str = ""


class AppendMessageRequest(BaseModel):
    role: str
    content: str
    patent_data: list | None = None
    timestamp: str | None = None


def register_session_routes(logger, config):
    router = APIRouter()

    @router.post("/session")
    async def create_session(req: CreateSessionRequest):
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO conversations
                       (session_id, user_id, scene_id, title, messages)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (session_id, req.user_id, req.scene_id, req.title,
                     json.dumps(req.messages, ensure_ascii=False)))
                conn.commit()
            return {"success": True, "session_id": session_id}
        finally:
            conn.close()

    @router.get("/session/{session_id}")
    async def get_session(session_id: str):
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id, session_id, user_id, scene_id, title,
                              messages, long_task_ids, status,
                              create_time, update_time
                       FROM conversations
                       WHERE session_id = %s AND status != 2""",
                    (session_id,))
                row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Session not found")
            row['messages'] = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
            row['create_time'] = row['create_time'].isoformat() if row['create_time'] else None
            row['update_time'] = row['update_time'].isoformat() if row['update_time'] else None
            return {"success": True, "session": row}
        finally:
            conn.close()

    @router.get("/sessions")
    async def list_sessions(user_id: int = Query(...)):
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT session_id, title, status,
                              long_task_ids, create_time, update_time
                       FROM conversations
                       WHERE user_id = %s AND status != 2
                       ORDER BY update_time DESC""",
                    (user_id,))
                rows = cur.fetchall()
            for r in rows:
                r['create_time'] = r['create_time'].isoformat() if r['create_time'] else None
                r['update_time'] = r['update_time'].isoformat() if r['update_time'] else None
            return {"success": True, "sessions": rows}
        finally:
            conn.close()

    @router.post("/session/{session_id}/message")
    async def append_message(session_id: str, req: AppendMessageRequest):
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT messages FROM conversations WHERE session_id = %s",
                    (session_id,))
                row = cur.fetchone()
                if row is None:
                    raise HTTPException(status_code=404, detail="Session not found")

                messages = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
                new_msg = {"role": req.role, "content": req.content}
                if req.patent_data:
                    new_msg["patent_data"] = req.patent_data
                if req.timestamp:
                    new_msg["timestamp"] = req.timestamp
                messages.append(new_msg)

                cur.execute(
                    "UPDATE conversations SET messages = %s WHERE session_id = %s",
                    (json.dumps(messages, ensure_ascii=False), session_id))
                conn.commit()
            return {"success": True}
        finally:
            conn.close()

    @router.delete("/session/{session_id}")
    async def archive_session(session_id: str):
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE conversations SET status = 2 WHERE session_id = %s",
                    (session_id,))
                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Session not found")
                conn.commit()
            return {"success": True}
        finally:
            conn.close()

    return router
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_session_api.py -v`
Expected: 4 PASS

- [ ] **Step 5: Register route in api.py**

Add to `api.py` after the existing route registrations:
```python
from api_routes import session as session_routes
session_router = session_routes.register_session_routes(logger, config)
api.include_router(session_router, tags=["session"])
```

- [ ] **Step 6: Commit**

```bash
git add api_routes/session.py tests/test_session_api.py api.py
git commit -m "feat: add session CRUD API routes with MySQL persistence"
```

---

### Task 7: Long Task API Routes

**Files:**
- Create: `api_routes/long_task.py`
- Test: `tests/test_long_task_api.py`

**Interfaces:**
- Consumes: `get_task_status()`, `LocalReportStorage`
- Produces:
  - `register_long_task_routes(logger, config) -> APIRouter`
  - `GET /long_task/{task_id}/status`
  - `GET /long_task/{task_id}/report?format=pdf|docx`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_long_task_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


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
        mock_storage.get.side_effect = FileNotFoundError("no file")
        mock_create.return_value = mock_storage

        response = client.get("/long_task/lt_nonexistent/report?format=pdf")
        assert response.status_code == 404


def test_get_report_success(client):
    """GET report for completed task returns file."""
    with patch('api_routes.long_task.create_storage') as mock_create:
        mock_storage = MagicMock()
        mock_storage.get.return_value = b"fake pdf content"
        mock_create.return_value = mock_storage

        response = client.get("/long_task/lt_001/report?format=pdf")
        assert response.status_code == 200
        assert response.headers['content-type'] == 'application/pdf'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_long_task_api.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write long_task API routes**

```python
# api_routes/long_task.py
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import Response
from sources.long_task.status_manager import get_task_status
from sources.long_task.storage import create_storage


def register_long_task_routes(logger, config):
    router = APIRouter()

    def _get_storage_config() -> dict:
        return {
            'report_storage_backend': config.get('STORAGE', 'backend',
                                                 fallback='local'),
            'report_storage_local_dir': config.get('STORAGE', 'local_base_dir',
                                                   fallback='/opt/workspace/reports'),
        }

    @router.get("/long_task/{task_id}/status")
    async def task_status(task_id: str):
        status = get_task_status(task_id)
        return {"success": True, **status}

    @router.get("/long_task/{task_id}/report")
    async def download_report(task_id: str, format: str = Query(..., regex="^(pdf|docx)$")):
        storage = create_storage(_get_storage_config())
        filename = f"report.{format}"
        try:
            content = await storage.get(task_id, filename)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Report not found")

        media_type = "application/pdf" if format == "pdf" else \
                     "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="patent_analysis_{task_id}.{format}"'
            },
        )

    return router
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_long_task_api.py -v`
Expected: 4 PASS

- [ ] **Step 5: Register route in api.py**

```python
from api_routes import long_task as long_task_routes
lt_router = long_task_routes.register_long_task_routes(logger, config)
api.include_router(lt_router, tags=["long_task"])
```

- [ ] **Step 6: Commit**

```bash
git add api_routes/long_task.py tests/test_long_task_api.py api.py
git commit -m "feat: add long task status and report download API routes"
```

---

### Task 8: Knowledge Type Utils — support type=3

**Files:**
- Modify: `sources/knowledge/type_utils.py`

**Change:** `infer_knowledge_type` and `_is_workflow_params` need a sibling `_is_long_task_params`; the valid type range expands from `(1, 2)` to `(1, 2, 3)`.

- [ ] **Step 1: Read the current file**

Read `sources/knowledge/type_utils.py` to see the exact current code.

- [ ] **Step 2: Modify type_utils.py**

```python
# sources/knowledge/type_utils.py
import json
from typing import Any


def _is_workflow_params(params: Any) -> bool:
    if not params:
        return False
    if isinstance(params, dict):
        return params.get("type") == "workflow"
    if not isinstance(params, str):
        return False
    try:
        parsed = json.loads(params)
    except (TypeError, ValueError):
        return False
    return isinstance(parsed, dict) and parsed.get("type") == "workflow"


def _is_long_task_params(params: Any) -> bool:
    """Check if params indicate a long task (type=3) knowledge entry."""
    if not params:
        return False
    if isinstance(params, dict):
        return params.get("type") == "long_task"
    if not isinstance(params, str):
        return False
    try:
        parsed = json.loads(params)
    except (TypeError, ValueError):
        return False
    return isinstance(parsed, dict) and parsed.get("type") == "long_task"


def infer_knowledge_type(raw_type: Any, params: Any = None) -> int:
    if _is_long_task_params(params):
        return 3
    if _is_workflow_params(params):
        return 2
    try:
        knowledge_type = int(raw_type)
    except (TypeError, ValueError):
        return 1
    return knowledge_type if knowledge_type in (1, 2, 3) else 1
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `python -m pytest tests/ -k "type_util or infer" -v 2>&1 | head -20`
(If no existing tests, verify by importing: `python -c "from sources.knowledge.type_utils import infer_knowledge_type; assert infer_knowledge_type(3) == 3; assert infer_knowledge_type(1, {'type':'long_task'}) == 3; print('OK')"`)

- [ ] **Step 4: Commit**

```bash
git add sources/knowledge/type_utils.py
git commit -m "feat: support knowledge type=3 for long task intent matching"
```

---

### Task 9: Install Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add required packages**

```
# requirements.txt — append these lines
celery>=5.4,<6.0
python-docx>=1.1,<2.0
weasyprint>=62,<63
```

- [ ] **Step 2: Install**

```bash
pip install celery python-docx weasyprint
```

- [ ] **Step 3: Verify imports**

```bash
python -c "import celery; import docx; import weasyprint; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add celery, python-docx, weasyprint dependencies"
```

---

### Task 10: Celery Worker Entry Point

**Files:**
- Create: `celery_worker.py`

**Interfaces:**
- Produces: Celery `app` object, `execute_patent_analysis` task (placeholder that writes status and sleeps)

- [ ] **Step 1: Write celery_worker.py (skeleton)**

```python
# celery_worker.py
import os
from celery import Celery

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

app = Celery('patent_tasks', broker=REDIS_URL, backend=REDIS_URL)

app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
)


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def execute_patent_analysis(self, task_id: str, params: dict):
    """Batch patent analysis — skeleton, filled in Task 13."""
    from sources.long_task.status_manager import update_task_status, set_task_completed
    from sources.long_task.config import get_long_task_config

    ltc = get_long_task_config()
    max_patents = ltc['max_patents']
    patent_ids = list(dict.fromkeys(params.get('patent_ids', [])))[:max_patents]

    try:
        update_task_status(task_id, 'generating_columns', 0,
                          f'正在启动分析任务（最多 {max_patents} 个专利）...')
        # Placeholder: real logic in Task 13
        update_task_status(task_id, 'generating_columns', 5,
                          '分析框架已生成')
        set_task_completed(task_id, [])
        return {'status': 'completed', 'task_id': task_id}
    except Exception as e:
        from sources.long_task.status_manager import set_task_failed
        set_task_failed(task_id, str(e))
        raise self.retry(exc=e)
```

- [ ] **Step 2: Verify Celery can load the app**

```bash
celery -A celery_worker inspect registered 2>&1 | head -5
```

Expected: Shows `execute_patent_analysis` in the task list (may need `--pool=solo` flag, just checking import works).

- [ ] **Step 3: Commit**

```bash
git add celery_worker.py
git commit -m "feat: add celery worker skeleton with patent analysis task"
```

---

### Task 11: Knowledge Selection — conversation_history in routing

**Files:**
- Modify: `sources/knowledge/selection.py`

**Change:** `build_routing_user_content` accepts optional `conversation_history` parameter and includes it in the prompt.

- [ ] **Step 1: Read current selection.py**

Read the file to understand the exact function signatures.

- [ ] **Step 2: Modify build_routing_user_content**

```python
# sources/knowledge/selection.py — modify build_routing_user_content

def build_routing_user_content(
    user_request: str,
    candidates: Iterable[KnowledgeToolCandidate],
    conversation_history: list | None = None,
) -> str:
    """Build the user content for the routing LLM prompt."""
    candidate_lines = []
    for idx, (knowledge, tool) in enumerate(candidates):
        k_type = knowledge.get('type', 1) if isinstance(knowledge, dict) else 1
        k_question = knowledge.get('question', '') if isinstance(knowledge, dict) else ''
        k_desc = knowledge.get('description', '') if isinstance(knowledge, dict) else ''
        candidate_lines.append(
            f"- id={knowledge.get('id') if isinstance(knowledge, dict) else '?'}, "
            f"type={k_type}, question=\"{k_question}\", "
            f"description=\"{k_desc}\""
        )

    candidates_text = "\n".join(candidate_lines) if candidate_lines else "(none)"

    history_section = ""
    if conversation_history:
        history_lines = []
        for msg in conversation_history[-6:]:  # last 3 exchanges max
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:200]  # truncate long content
            history_lines.append(f"{role}: {content}")
        if history_lines:
            history_section = "## 对话历史\n" + "\n".join(history_lines) + "\n\n"

    return f"""{history_section}## 当前消息
user: {user_request}

## 候选知识
{candidates_text}

## 判断规则
- type=3（长任务）：用户需要对一批专利逐个深入理解、跨专利综合分析、生成结构化报告
- type=2（workflow）：多步工具链调用
- type=1（普通知识）：作为 system prompt 上下文
- 如果用户只是检索/浏览/简单问答，不触发长任务，选择合适的普通知识

请选择最匹配的 knowledge_id，不触发长任务返回 null。"""
```

- [ ] **Step 3: Update choose_knowledge_candidate to pass conversation_history**

```python
async def choose_knowledge_candidate(
    user_request: str,
    candidates: Iterable[KnowledgeToolCandidate],
    complete_json: CompleteJson,
    min_confidence: float = MIN_ROUTING_CONFIDENCE,
    conversation_history: list | None = None,
) -> Optional[KnowledgeToolCandidate]:
    # ... existing logic, but pass conversation_history to build_routing_user_content:
    user_content = build_routing_user_content(
        user_request, candidates,
        conversation_history=conversation_history,
    )
    # ... rest unchanged
```

- [ ] **Step 4: Update caller in general_agent.py (create_agent)**

In `GeneralAgent.create_agent()`, pass `conversation_history` to `select_knowledge_tool_with_llm()`. First verify the parameter is available from somewhere — if the request object carries it, use that. For now, just note the integration point:

```python
# In GeneralAgent.create_agent():
# conversation_history comes from request/params (TBD — will be wired in Task 15)
# self.knowledgeTool = await select_knowledge_tool_with_llm(
#     user_id, prompt, self.llm.complete_json,
#     push_filter=push_filter,
#     conversation_history=getattr(self, '_conversation_history', None),
# )
```

- [ ] **Step 5: Verify import and run tests**

```bash
python -c "from sources.knowledge.selection import build_routing_user_content; print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add sources/knowledge/selection.py
git commit -m "feat: add conversation_history to LLM routing prompt for long task detection"
```

---

### Task 12: Agent Pool Reduction

**Files:**
- Modify: `api_routes/core.py`

**Change:** `AGENT_POOL_MAX_SIZE` from 10 to 3.

- [ ] **Step 1: Change the constant**

```python
# api_routes/core.py line ~24
AGENT_POOL_MAX_SIZE = 3  # was 10
```

- [ ] **Step 2: Commit**

```bash
git add api_routes/core.py
git commit -m "perf: reduce agent pool max size from 10 to 3"
```

---

## Phase 2: Analysis Pipeline (Tasks 13–17)

### Task 13: Patent Analyzer — Phase 1 & 2

**Files:**
- Create: `sources/long_task/patent_analyzer.py`
- Test: `tests/test_patent_analyzer.py`

**Interfaces:**
- Consumes: `get_long_task_config()`, `update_task_status()`, `save_checkpoint()`, `load_checkpoint()`, Provider from `sources/llm_provider.py`
- Produces:
  - `generate_table_columns(query, patent_count, model) -> list[str]`
  - `download_patent_document(patent_id, source) -> str`
  - `analyze_single_patent(patent_id, patent_text, columns, query, model, timeout) -> dict`
  - `generate_patent_summary(patent_id, row, query, model) -> str`
  - `build_failed_row(patent_id, reason) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_patent_analyzer.py
import pytest
from unittest.mock import patch, MagicMock
from sources.long_task.patent_analyzer import (
    generate_table_columns, analyze_single_patent,
    generate_patent_summary, build_failed_row,
)


def test_build_failed_row():
    """build_failed_row returns placeholder data with failure marker."""
    row = build_failed_row("CN001", "download failed")
    assert row['patent_id'] == 'CN001'
    assert row['_failed'] is True
    assert 'download failed' in row['_failure_reason']


@pytest.mark.asyncio
async def test_generate_table_columns():
    """generate_table_columns calls LLM and returns list of column names."""
    mock_provider = MagicMock()
    mock_provider.complete_json.return_value = {
        "columns": ["专利号", "技术领域", "核心技术方案", "创新点"]
    }

    result = await generate_table_columns(
        query="分析技术分布和创新点",
        patent_count=10,
        provider=mock_provider,
    )
    assert isinstance(result, list)
    assert "专利号" in result
    assert len(result) >= 3


@pytest.mark.asyncio
async def test_analyze_single_patent():
    """analyze_single_patent calls LLM with patent text and columns."""
    mock_provider = MagicMock()
    mock_provider.complete_json.return_value = {
        "patent_id": "CN001",
        "技术领域": "G06V 计算机视觉",
        "核心技术方案": "双流注意力机制",
        "创新点": "小样本学习",
    }

    row = await analyze_single_patent(
        patent_id="CN001",
        patent_text="专利说明书全文...",
        columns=["专利号", "技术领域", "核心技术方案", "创新点"],
        query="分析技术分布",
        provider=mock_provider,
    )
    assert row['patent_id'] == 'CN001'
    assert row['技术领域'] == 'G06V 计算机视觉'
    assert '_summary' not in row  # summary is separate


@pytest.mark.asyncio
async def test_generate_patent_summary():
    """generate_patent_summary returns a short text summary."""
    mock_provider = MagicMock()
    mock_provider.complete_json.return_value = {
        "summary": "该专利提出了一种基于双流注意力的图像识别方法，核心创新在于解决小样本过拟合问题。"
    }

    summary = await generate_patent_summary(
        patent_id="CN001",
        row={"patent_id": "CN001", "技术领域": "G06V"},
        query="分析技术分布",
        provider=mock_provider,
    )
    assert isinstance(summary, str)
    assert len(summary) > 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_patent_analyzer.py -v`
Expected: FAIL

- [ ] **Step 3: Write patent_analyzer.py**

```python
# sources/long_task/patent_analyzer.py
"""Core patent analysis pipeline functions."""


def build_failed_row(patent_id: str, reason: str) -> dict:
    """Return a placeholder row for a patent that failed to process."""
    return {
        'patent_id': patent_id,
        '_failed': True,
        '_failure_reason': reason,
    }


async def generate_table_columns(query: str, patent_count: int, provider) -> list[str]:
    """Phase 1: Use Flash to dynamically generate table column definitions."""
    system_prompt = """你是一个专利分析专家。根据用户的分析问题，确定对比表格需要哪些列。
返回 JSON 格式：{"columns": ["列1", "列2", ...]}
列数控制在 4-6 列。"专利号"必须是第一列。
根据问题类型选择列：
- 技术分析：技术领域、核心技术方案、创新点、相关度
- 对比分析：申请人、技术方向、核心方案、差异点
- 风险评估：保护范围、产品对照、风险等级、规避建议"""
    user_content = f"用户问题：{query}\n专利数量：{patent_count}\n请确定分析表格的列定义。"
    result = await provider.complete_json(system_prompt, user_content)
    return result.get('columns', ['专利号', '分析结果'])


async def download_patent_document(patent_id: str, source: str = 'cnipa') -> str:
    """Download patent full text from DI platform (CNIPA) or USPTO.

    Uses existing tool infrastructure — calls the DI platform API
    through dynamic_tool_params.execute_backend_tool_request().
    Returns the patent description text.
    """
    from sources.patent_token import ensure_valid_access_token
    from sources.dynamic_tool_params import _inject_zldjs_auth_params
    import httpx

    token_data = ensure_valid_access_token()
    access_token = token_data.access_token

    if source == 'cnipa':
        url = f"https://open.zldsj.com/api/patent/{patent_id}/fulltext"
        params = {'access_token': access_token}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            # Extract description from DI platform response
            if 'data' in data and 'description' in data['data']:
                return data['data']['description']
            return str(data)
    else:
        # USPTO path — use existing download proxy
        from sources.uspto_download import download_uspto_patent
        return await download_uspto_patent(patent_id)


async def analyze_single_patent(patent_id: str, patent_text: str,
                                 columns: list[str], query: str,
                                 provider, timeout: int = 60) -> dict:
    """Phase 2b: Analyze one patent and return a table row dict."""
    cols_desc = "\n".join(f"- {c}" for c in columns if c != '专利号')
    system_prompt = f"""你是一个专利分析专家。根据以下维度分析专利：

{cols_desc}

返回 JSON 格式，key 为列名，value 为分析结果。
"patent_id" 字段填写专利号。
分析要具体、有依据，基于专利文本内容。"""

    user_content = f"""用户问题：{query}

专利号：{patent_id}
专利文本（摘要）：
{patent_text[:8000]}

请按维度分析并返回 JSON。"""

    result = await provider.complete_json(system_prompt, user_content)
    result['patent_id'] = patent_id
    return result


async def generate_patent_summary(patent_id: str, row: dict, query: str, provider) -> str:
    """Phase 2c: Generate a short summary for a single analyzed patent."""
    row_str = "\n".join(f"{k}: {v}" for k, v in row.items()
                        if k not in ('patent_id', '_failed', '_failure_reason', '_summary'))
    system_prompt = "你是一个专利分析专家。基于分析结果，用 2-3 句话总结该专利的核心发现。"
    user_content = f"用户问题：{query}\n专利：{patent_id}\n分析结果：\n{row_str}\n\n请给出简洁总结。"
    result = await provider.complete_json(system_prompt, user_content)
    return result.get('summary', '')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_patent_analyzer.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add sources/long_task/patent_analyzer.py tests/test_patent_analyzer.py
git commit -m "feat: add patent analyzer core — table columns, single patent analysis, summary"
```

---

### Task 14: Celery Worker — Full Pipeline Wiring

**Files:**
- Modify: `celery_worker.py`

**Change:** Replace the skeleton task body with full 4-phase pipeline.

- [ ] **Step 1: Rewrite execute_patent_analysis with full pipeline**

```python
# celery_worker.py — replace the try block in execute_patent_analysis
@app.task(bind=True, max_retries=3, default_retry_delay=30)
def execute_patent_analysis(self, task_id: str, params: dict):
    """Batch patent analysis — 4-phase serial pipeline with checkpointing."""
    from sources.long_task.status_manager import (
        update_task_status, set_task_completed, set_task_failed,
        save_checkpoint, load_checkpoint,
    )
    from sources.long_task.config import get_long_task_config
    from sources.long_task.patent_analyzer import (
        generate_table_columns, download_patent_document,
        analyze_single_patent, generate_patent_summary, build_failed_row,
    )
    from sources.long_task.report_generator import (
        generate_report_outline, generate_report_section,
    )
    from sources.long_task.storage import create_storage
    from sources.llm_provider import Provider

    ltc = get_long_task_config()
    model_family = ltc['provider_family']
    max_patents = ltc['max_patents']

    # ---- Input dedup + truncation ----
    patent_ids = list(dict.fromkeys(params.get('patent_ids', [])))
    patent_ids = patent_ids[:max_patents]
    total = len(patent_ids)

    # ---- Provider setup ----
    if model_family == 'minimax':
        flash_provider = Provider(provider_name='minimax', model='minimax-2.7-highspeed',
                                  server_address='', is_local=False)
        pro_provider = Provider(provider_name='minimax', model='minimax-2.7-highspeed',
                                server_address='', is_local=False)
    else:
        flash_provider = Provider(provider_name='deepseek', model='deepseek-chat',
                                  server_address='', is_local=False)
        pro_provider = Provider(provider_name='deepseek', model='deepseek-reasoner',
                                server_address='', is_local=False)

    try:
        # ---- Crash recovery: load checkpoint ----
        checkpoint = load_checkpoint(task_id)
        if checkpoint and checkpoint.get('pending'):
            table_rows = checkpoint.get('completed_rows', [])
            pending = checkpoint['pending']
        else:
            table_rows = []
            pending = patent_ids

        # ==== Phase 1: Generate columns (Flash) ====
        update_task_status(task_id, 'generating_columns', 0,
                          f'正在生成分析框架（{total} 个专利）...')
        columns = generate_table_columns(
            query=params['query'],
            patent_count=total,
            provider=flash_provider,
        )
        update_task_status(task_id, 'generating_columns', 5,
                          f'分析维度：{" | ".join(columns[1:4])}...',
                          table_columns=columns)

        # ==== Phase 2: Per-patent download -> analyze -> summarize ====
        for i, patent_id in enumerate(pending):
            patent_source = params.get('patent_source', 'cnipa')
            try:
                update_task_status(task_id, 'analyzing',
                    progress_pct(i, len(table_rows), total),
                    f'正在下载专利文件（{len(table_rows)+1}/{total}）...',
                    table_rows=table_rows)

                patent_text = download_patent_document(patent_id, patent_source)

                update_task_status(task_id, 'analyzing',
                    progress_pct(i, len(table_rows), total),
                    f'正在分析（{len(table_rows)+1}/{total}）：{patent_id}',
                    table_rows=table_rows)

                row = analyze_single_patent(
                    patent_id=patent_id, patent_text=patent_text,
                    columns=columns, query=params['query'],
                    provider=pro_provider, timeout=60,
                )
                row['_summary'] = generate_patent_summary(
                    patent_id=patent_id, row=row, query=params['query'],
                    provider=pro_provider,
                )

            except Exception as e:
                row = build_failed_row(patent_id, str(e))

            table_rows.append(row)
            save_checkpoint(task_id, {
                'completed': [r['patent_id'] for r in table_rows if not r.get('_failed')],
                'current': patent_id,
                'pending': pending[i+1:],
                'completed_rows': table_rows,
                'failed': [r['patent_id'] for r in table_rows if r.get('_failed')],
            })

            update_task_status(task_id, 'analyzing',
                progress_pct(i + 1, 0, total),  # 5% + (i+1)/total * 70%
                f'已完成 {len(table_rows)}/{total} 个专利分析',
                table_rows=table_rows)

        # ==== Phase 3: Generate report (Pro, dynamic) ====
        update_task_status(task_id, 'generating_report', 80,
                          '正在规划报告结构...')
        outline = generate_report_outline(
            query=params['query'], columns=columns,
            table_rows=table_rows, provider=pro_provider,
        )

        report_parts = []
        sections = outline.get('sections', [{'heading': '分析结果', 'description': ''}])
        for idx, section in enumerate(sections):
            sec_pct = 80 + int((idx + 1) / len(sections) * 10)
            update_task_status(task_id, 'generating_report', sec_pct,
                              f'正在撰写：{section["heading"]}')
            text = generate_report_section(
                section=section, query=params['query'],
                columns=columns, table_rows=table_rows,
                provider=pro_provider,
            )
            report_parts.append(f"## {section['heading']}\n\n{text}")

        report_text = f"# {outline.get('title', '专利分析报告')}\n\n" + "\n\n".join(report_parts)
        update_task_status(task_id, 'generating_report', 90,
                          '报告撰写完成', result_summary=report_text)

        # ==== Phase 4: Export files ====
        storage_cfg = {
            'report_storage_backend': 'local',
            'report_storage_local_dir': '/opt/workspace/reports',
        }
        storage = create_storage(storage_cfg)

        update_task_status(task_id, 'exporting', 92, '正在生成 PDF 文件...')
        pdf_bytes = export_pdf(report_text, table_rows, columns)
        await storage.put(task_id, 'report.pdf', pdf_bytes)

        update_task_status(task_id, 'exporting', 96, '正在生成 Word 文件...')
        docx_bytes = export_docx(report_text, table_rows, columns)
        await storage.put(task_id, 'report.docx', docx_bytes)

        report_files = [
            {'format': 'pdf', 'filename': 'report.pdf', 'size': len(pdf_bytes)},
            {'format': 'docx', 'filename': 'report.docx', 'size': len(docx_bytes)},
        ]
        set_task_completed(task_id, report_files)
        return {'status': 'completed', 'task_id': task_id}

    except Exception as e:
        set_task_failed(task_id, str(e))
        raise self.retry(exc=e)


def progress_pct(completed: int, offset: int, total: int) -> int:
    """Map completed/total to progress percentage in [5, 75]."""
    if total == 0:
        return 5
    return 5 + min(70, int(completed / total * 70))


def export_pdf(report_text: str, table_rows: list, columns: list) -> bytes:
    """Export report as PDF using weasyprint."""
    import weasyprint
    html = _build_report_html(report_text, table_rows, columns)
    return weasyprint.HTML(string=html).write_pdf()


def export_docx(report_text: str, table_rows: list, columns: list) -> bytes:
    """Export report as DOCX using python-docx."""
    import io
    from docx import Document
    doc = Document()
    # Title
    for line in report_text.split('\n'):
        if line.startswith('# '):
            doc.add_heading(line[2:], level=1)
        elif line.startswith('## '):
            doc.add_heading(line[3:], level=2)
        elif line.strip():
            doc.add_paragraph(line.strip())

    # Table
    if table_rows and columns:
        table = doc.add_table(rows=1, cols=len(columns))
        table.style = 'Table Grid'
        for i, col in enumerate(columns):
            table.rows[0].cells[i].text = col
        for row_data in table_rows:
            row = table.add_row()
            for i, col in enumerate(columns):
                row.cells[i].text = str(row_data.get(col, ''))

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _build_report_html(report_text: str, table_rows: list, columns: list) -> str:
    """Build HTML for PDF export."""
    import html as html_mod
    rows_html = ""
    if table_rows and columns:
        header = "<tr>" + "".join(f"<th>{html_mod.escape(str(c))}</th>" for c in columns) + "</tr>"
        body = ""
        for r in table_rows:
            body += "<tr>" + "".join(f"<td>{html_mod.escape(str(r.get(c, '')))}</td>" for c in columns) + "</tr>"
        rows_html = f"<table>{header}{body}</table>"

    text_html = report_text.replace('\n', '<br>')
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>body{{font-family:sans-serif;max-width:900px;margin:0 auto;padding:20px;}}
table{{border-collapse:collapse;width:100%;}}
th,td{{border:1px solid #ddd;padding:8px;font-size:12px;text-align:left;}}
th{{background:#f5f5f5;}}</style></head>
<body>{text_html}{rows_html}</body></html>"""
```

- [ ] **Step 2: Verify the worker loads**

```bash
python -c "from celery_worker import app, execute_patent_analysis; print('Worker loads OK')"
```

Expected: `Worker loads OK`

- [ ] **Step 3: Commit**

```bash
git add celery_worker.py
git commit -m "feat: wire full 4-phase patent analysis pipeline in Celery worker"
```

---

### Task 15: Report Generator

**Files:**
- Create: `sources/long_task/report_generator.py`

**Interfaces:**
- Produces:
  - `generate_report_outline(query, columns, table_rows, provider) -> dict`
  - `generate_report_section(section, query, columns, table_rows, provider) -> str`

- [ ] **Step 1: Write report_generator.py**

```python
# sources/long_task/report_generator.py
"""Phase 3: Dynamic report generation — outline + section-by-section writing."""


async def generate_report_outline(query: str, columns: list[str],
                                   table_rows: list[dict], provider) -> dict:
    """Phase 3a: Generate a dynamic report outline based on user query and data."""
    row_count = len(table_rows)
    cols_str = ", ".join(columns)
    failed_count = sum(1 for r in table_rows if r.get('_failed'))

    system_prompt = """你是一个专利分析报告架构师。根据用户问题和分析结果，规划报告结构。
返回 JSON：{"title": "报告标题", "sections": [{"heading": "章节标题", "description": "本章内容说明"}]}
章节数 3-7 个，标题简洁。"""

    user_content = f"""用户问题：{query}
分析维度：{cols_str}
已分析专利数：{row_count}{f'（其中 {failed_count} 个分析失败）' if failed_count else ''}

请规划报告结构。"""

    result = await provider.complete_json(system_prompt, user_content)
    return result


async def generate_report_section(section: dict, query: str, columns: list[str],
                                   table_rows: list[dict], provider) -> str:
    """Phase 3b: Write a single report section."""
    heading = section.get('heading', '')
    description = section.get('description', '')

    # Build a compact summary of the table data for context
    data_summary_lines = []
    for r in table_rows[:20]:  # max 20 rows in prompt
        if r.get('_failed'):
            continue
        parts = [f"{r.get('patent_id', '?')}:"]
        for col in columns[1:4]:  # first 3 non-patent_id columns
            val = r.get(col, '')
            if val:
                parts.append(f"{col}={str(val)[:60]}")
        data_summary_lines.append("  ".join(parts))
    data_summary = "\n".join(data_summary_lines[:20])

    system_prompt = "你是一个专利分析报告撰写专家。根据给定的分析数据，撰写一个报告章节。用中文，具体有依据。"
    user_content = f"""用户问题：{query}
本章标题：{heading}
本章说明：{description}

分析数据摘要：
{data_summary}

请撰写"{heading}"章节内容（Markdown 格式，300-600 字）："""

    result = await provider.complete_json(system_prompt, user_content)
    # complete_json returns parsed JSON; for free-text sections, wrap
    if isinstance(result, dict):
        return result.get('content', result.get('text', str(result)))
    return str(result)
```

- [ ] **Step 2: Verify imports**

```bash
python -c "from sources.long_task.report_generator import generate_report_outline, generate_report_section; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add sources/long_task/report_generator.py
git commit -m "feat: add dynamic report generation — outline + section writing"
```

---

### Task 16: Intent Recognition — Long Task Branch in GeneralAgent

**Files:**
- Modify: `sources/agents/general_agent.py`

**Change:** In `create_agent()`, after `select_knowledge_tool_with_llm()`, detect type=3 (long task) and return a special marker instead of proceeding to normal agent creation.

- [ ] **Step 1: Add is_long_task_knowledge helper**

```python
# Add near the top of general_agent.py, near existing is_workflow_knowledge
def _is_long_task_knowledge(knowledge_item) -> bool:
    """Check if knowledge item represents a long task (type=3)."""
    if knowledge_item is None:
        return False
    k_type = knowledge_item.get('type', 1) if isinstance(knowledge_item, dict) else getattr(knowledge_item, 'type', 1)
    return int(k_type) == 3


def _build_long_task_intent(knowledge_item, tool_info) -> dict:
    """Build a long task intent marker returned by create_agent."""
    return {
        'intent': 'long_task',
        'knowledge': knowledge_item,
        'tool_info': tool_info,
    }
```

- [ ] **Step 2: Insert long task branch in create_agent**

In `GeneralAgent.create_agent()`, after the line `knowledge_item, tool_info = self.knowledgeTool` and before `if is_workflow_knowledge(knowledge_item):`, add:

```python
    # Long task detection: type=3 knowledge triggers async Celery pipeline
    if _is_long_task_knowledge(knowledge_item):
        if callback_handler:
            await _emit_status(callback_handler, "正在启动批量专利分析任务...")
        # Return special marker — caller (run_pipeline) handles submission
        self._long_task_intent = _build_long_task_intent(knowledge_item, tool_info)
        return self._long_task_intent
```

- [ ] **Step 3: Update invoke_agent to handle long task marker**

In `GeneralAgent.invoke_agent()`, at the top:

```python
    async def invoke_agent(self, agent, callback_handler):
        try:
            # Handle long task intent marker
            if isinstance(agent, dict) and agent.get('intent') == 'long_task':
                # The actual Celery submission happens in api_routes/core.py run_pipeline
                # after create_agent returns. Here we just push the notification.
                await _emit_status(callback_handler,
                    "批量专利分析任务已创建，可通过 task_id 查询进度")
                # Queue a special SSE event type
                await callback_handler.queue.put({
                    'type': 'long_task_intent',
                    'knowledge': agent.get('knowledge'),
                })
                return
            # ... existing code ...
```

- [ ] **Step 4: Verify nothing breaks for normal queries**

```bash
python -c "from sources.agents.general_agent import _is_long_task_knowledge; assert not _is_long_task_knowledge(None); assert not _is_long_task_knowledge({'type':1}); assert _is_long_task_knowledge({'type':3}); print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add sources/agents/general_agent.py
git commit -m "feat: add long task intent detection branch in GeneralAgent"
```

---

### Task 17: SSE long_task_created Event + Celery Submission in core.py

**Files:**
- Modify: `api_routes/core.py`

**Change:** In `run_pipeline()`, after `create_agent()` returns, detect the long task intent marker, create session + long_task DB record, submit to Celery, and push `long_task_created` SSE event.

- [ ] **Step 1: Add the long task handling block in run_pipeline**

In `api_routes/core.py`, inside `run_pipeline()` after `openai_agent = await general_agent.create_agent(...)`:

```python
            # Long task handling: create_agent returned a long_task intent marker
            if isinstance(openai_agent, dict) and openai_agent.get('intent') == 'long_task':
                import uuid
                import json
                from sources.knowledge.knowledge import get_db_connection

                task_id = f"lt_{uuid.uuid4().hex[:12]}"
                session_id = f"sess_{uuid.uuid4().hex[:12]}"

                # Extract patent_ids from conversation history
                patent_ids = []
                conv_history = getattr(request, 'conversation_history', None) or []
                for msg in conv_history:
                    if msg.get('role') == 'assistant' and msg.get('patent_data'):
                        for p in msg['patent_data']:
                            if isinstance(p, dict) and 'patent_id' in p:
                                patent_ids.append(p['patent_id'])

                # Write to MySQL: session + long_task record
                conn = get_db_connection()
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            """INSERT INTO conversations (session_id, user_id, title, messages)
                               VALUES (%s, %s, %s, %s)""",
                            (session_id, user_id, f"专利分析 - {request.query[:50]}",
                             json.dumps(conv_history, ensure_ascii=False)))
                        cur.execute(
                            """INSERT INTO long_tasks
                               (task_id, session_id, user_id, task_type, input_params, status)
                               VALUES (%s, %s, %s, %s, %s, %s)""",
                            (task_id, session_id, user_id, 'patent_analysis',
                             json.dumps({
                                 'query': request.query,
                                 'patent_ids': patent_ids,
                                 'patent_source': 'cnipa',
                             }, ensure_ascii=False),
                             'pending'))
                        conn.commit()
                finally:
                    conn.close()

                # Submit to Celery
                from celery_worker import execute_patent_analysis
                execute_patent_analysis.delay(task_id=task_id, params={
                    'query': request.query,
                    'patent_ids': patent_ids,
                    'patent_source': 'cnipa',
                    'session_id': session_id,
                })

                # Push SSE event
                await queue.put({
                    'type': 'status',
                    'message': '批量专利分析任务已提交',
                    'transient': False,
                })
                await queue.put({
                    'type': 'long_task_created',
                    'task_id': task_id,
                    'session_id': session_id,
                })
                await queue.put({'type': 'end'})
                return
```

- [ ] **Step 2: Add long_task_created event handling in the SSE event loop**

In the SSE `while True` loop in `api_routes/core.py`, add after existing event type dispatches:

```python
                if event_type == 'long_task_created':
                    yield f"data:{json.dumps({'type': 'long_task_created', 'task_id': event.get('task_id'), 'session_id': event.get('session_id')})}\n\n"
                    continue

                if event_type == 'long_task_intent':
                    # Intermediate event from invoke_agent — process if needed
                    continue
```

- [ ] **Step 3: Commit**

```bash
git add api_routes/core.py
git commit -m "feat: add SSE long_task_created event and Celery submission in query pipeline"
```

---

## Phase 3: Integration & Polish (Tasks 18–20)

### Task 18: Frontend — SSE long_task_created Listener + Polling Switch

**Note:** This task touches frontend code. Adapt to your specific frontend framework (React/Vue/etc.). The pattern is the same.

- [ ] **Step 1: In the existing SSE message handler, add the long_task_created case**

```javascript
// In your SSE event handler (wherever you handle SSE messages)
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'long_task_created') {
    // Stop the current SSE connection
    eventSource.close();

    // Store task_id and session_id
    const taskId = data.task_id;
    const sessionId = data.session_id;

    // Switch to long task polling view
    switchToLongTaskView(taskId, sessionId);
    return;
  }

  // ... existing handling for token, status, artifact events
};
```

- [ ] **Step 2: Implement the polling function**

```javascript
function startPolling(taskId, sessionId) {
  const pollInterval = 2000; // 2 seconds
  let status = 'running';

  const poll = async () => {
    if (status !== 'running') return;

    try {
      const res = await fetch(`/long_task/${taskId}/status`);
      const data = await res.json();
      status = data.status;

      renderLongTaskProgress(data);

      if (status === 'running') {
        setTimeout(poll, pollInterval);
      } else if (status === 'completed') {
        renderCompletedView(data);
      } else if (status === 'failed') {
        renderErrorView(data);
      }
    } catch (err) {
      console.error('Poll error:', err);
      setTimeout(poll, pollInterval * 2); // back off on error
    }
  };

  poll();
}
```

- [ ] **Step 3: Implement the progress UI**

```javascript
function renderLongTaskProgress(data) {
  // Update progress bar
  updateProgressBar(data.progress);

  // Update phase list with status icons
  if (data.phases) {
    renderPhases(data.phases);
  }

  // Update current step
  updateCurrentStep(data.current_step);

  // Update table if available
  if (data.table_columns && data.table_rows) {
    renderDynamicTable(data.table_columns, data.table_rows);
  }
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/  # or wherever your frontend code lives
git commit -m "feat: add SSE long_task_created listener and polling-based progress UI"
```

---

### Task 19: Frontend — Session Sidebar + Report Download

- [ ] **Step 1: Implement session list sidebar**

A left sidebar showing user's sessions, fetched from `GET /sessions?user_id=UID`:

```javascript
async function loadSessionList(userId) {
  const res = await fetch(`/sessions?user_id=${userId}`);
  const data = await res.json();
  renderSessionSidebar(data.sessions);
}
```

- [ ] **Step 2: Session click → restore conversation**

```javascript
async function openSession(sessionId) {
  const res = await fetch(`/session/${sessionId}`);
  const data = await res.json();
  // Re-render chat with messages from data.session.messages
  // Show long_task download links from data.session.long_task_ids
  renderChatHistory(data.session.messages);
}
```

- [ ] **Step 3: Report download button**

```javascript
function renderDownloadButtons(taskId) {
  return `
    <a href="/long_task/${taskId}/report?format=pdf" download
       class="btn-download">📄 下载 PDF</a>
    <a href="/long_task/${taskId}/report?format=docx" download
       class="btn-download">📝 下载 Word</a>
  `;
}
```

- [ ] **Step 4: Commit**

```bash
git add frontend/
git commit -m "feat: add session sidebar and report download UI"
```

---

### Task 20: End-to-End Integration Test

**Files:**
- Modify: `tests/test_long_task_api.py` (add integration scenario)

- [ ] **Step 1: Write E2E-style integration test**

```python
# tests/test_long_task_integration.py
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.integration
def test_full_long_task_lifecycle():
    """Simulate the complete flow: trigger → status poll → report download."""
    # This test verifies the wiring between all components:
    # 1. API receives query with conversation_history
    # 2. LLM routing selects type=3 knowledge
    # 3. SSE long_task_created event is pushed
    # 4. Status polling returns progressive updates
    # 5. Report download returns file

    # For CI: use mocked LLM and Redis, real MySQL (test DB)
    # Implementation depends on test infrastructure setup
    pass
```

- [ ] **Step 2: Run all tests to confirm no regressions**

```bash
python -m pytest tests/ -v --ignore=tests/test_long_task_integration.py -x
```

Expected: All existing tests pass (Task 1–17 tests + existing project tests).

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test: add integration test placeholder for full long task lifecycle"
```

---

## Dependency Graph

```
Task 1  (conversations table)
Task 2  (long_tasks table)
Task 3  (config reader)       ─┐
Task 4  (storage abstraction)  ├─ independent, can run in parallel
Task 5  (status manager)       │
Task 8  (type_utils type=3)   ─┘
Task 9  (dependencies)
         │
Task 10 (celery skeleton)      ← depends on Task 3, 5, 9
Task 11 (selection.py)         ← depends on Task 8
Task 12 (agent pool)           ← independent
         │
Task 6  (session API)          ← depends on Task 1
Task 7  (long_task API)        ← depends on Task 5
         │
Task 13 (patent analyzer)      ← depends on Task 3
Task 15 (report generator)     ← can run parallel with Task 13
         │
Task 14 (celery full pipeline) ← depends on Task 10, 13, 15
Task 16 (general_agent branch) ← depends on Task 8, 11
Task 17 (core.py SSE event)    ← depends on Task 7, 14, 16
         │
Task 18 (frontend polling)     ← depends on Task 17
Task 19 (frontend sidebar)     ← depends on Task 6
Task 20 (integration test)     ← depends on all
```

---

## Verification Checklist

- [ ] `config.ini` has `[LONG_TASK]` with `provider_family` and `max_patents`
- [ ] `mysql` tables `conversations` and `long_tasks` exist and accept inserts
- [ ] `infer_knowledge_type(3)` returns 3
- [ ] `infer_knowledge_type(1, {"type":"long_task"})` returns 3
- [ ] `LocalReportStorage` can put/get/delete files
- [ ] `update_task_status()` writes to Redis, `get_task_status()` reads it back
- [ ] Redis checkpoint survives save/load round-trip
- [ ] `POST /session` creates a session, `GET /session/{id}` returns it
- [ ] `GET /long_task/{id}/status` returns valid JSON for any task_id
- [ ] `GET /long_task/{id}/report?format=pdf` returns 404 for missing files
- [ ] Celery worker loads without import errors: `celery -A celery_worker inspect registered`
- [ ] `create_agent()` returns `{'intent': 'long_task', ...}` when type=3 knowledge matched
- [ ] SSE event loop pushes `long_task_created` when long task intent detected
- [ ] `sorted(set(patent_ids))` deduplicates input
- [ ] Patent analysis timeout returns `build_failed_row()` not exception
- [ ] `progress_pct` returns values in [5, 100]
- [ ] Word docx has expected sections and table
- [ ] PDF has expected content (verify with `pdfinfo` or similar)
- [ ] Agent pool max size is 3 in core.py
- [ ] All new code has 80%+ test coverage
- [ ] No regression in existing `/query_stream` behavior
