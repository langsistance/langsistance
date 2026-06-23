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
