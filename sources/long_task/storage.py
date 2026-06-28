"""Abstract interface for report file storage."""

import configparser
import os
from abc import ABC, abstractmethod


def get_storage_config(config_path: str = "config.ini") -> dict:
    """Read [STORAGE] section from config file, with env-var overrides.

    Returns a dict suitable for ``create_storage()``.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    return {
        "report_storage_backend": cfg.get("STORAGE", "backend",
                                           fallback="local"),
        "report_storage_local_dir": cfg.get("STORAGE", "local_base_dir",
                                             fallback="/opt/workspace/reports"),
        "report_storage_cos_bucket": os.getenv(
            "COS_REPORT_BUCKET",
            cfg.get("STORAGE", "cos_bucket", fallback=""),
        ),
        "report_storage_cos_region": os.getenv(
            "COS_REGION",
            cfg.get("STORAGE", "cos_region", fallback="ap-hongkong"),
        ),
        "report_storage_cos_secret_id": os.getenv(
            "COS_SECRET_ID",
            cfg.get("STORAGE", "cos_secret_id", fallback=""),
        ),
        "report_storage_cos_secret_key": os.getenv(
            "COS_SECRET_KEY",
            cfg.get("STORAGE", "cos_secret_key", fallback=""),
        ),
        "report_storage_cos_prefix": os.getenv(
            "COS_REPORT_PREFIX",
            cfg.get("STORAGE", "cos_prefix", fallback="reports"),
        ),
    }


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
    if backend == 'cos':
        from sources.long_task.cos_storage import COSReportStorage
        return COSReportStorage(
            bucket=config['report_storage_cos_bucket'],
            region=config.get('report_storage_cos_region', 'ap-hongkong'),
            secret_id=config['report_storage_cos_secret_id'],
            secret_key=config['report_storage_cos_secret_key'],
            prefix=config.get('report_storage_cos_prefix', 'reports'),
        )
    raise ValueError(f"Unknown storage backend: {backend}")
