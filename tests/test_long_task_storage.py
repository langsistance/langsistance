import os
import tempfile
import unittest
import shutil

from sources.long_task.storage import LocalReportStorage, create_storage


class TestLocalReportStorage(unittest.IsolatedAsyncioTestCase):
    """Tests for LocalReportStorage."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage = LocalReportStorage(base_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_put_and_get(self):
        """Should store and retrieve file content."""
        task_id = "lt_test_001"
        content = b"fake report content"

        path = await self.storage.put(task_id, "report.pdf", content)
        retrieved = await self.storage.get(task_id, "report.pdf")

        self.assertEqual(retrieved, content)
        self.assertTrue(os.path.exists(path))

    async def test_delete_removes_all_files(self):
        """Should remove all files for a task."""
        task_id = "lt_test_002"

        await self.storage.put(task_id, "report.pdf", b"pdf content")
        await self.storage.put(task_id, "report.docx", b"docx content")

        await self.storage.delete(task_id)

        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, task_id)))

    async def test_get_nonexistent_raises(self):
        """Should raise FileNotFoundError for missing files."""
        with self.assertRaises(FileNotFoundError):
            await self.storage.get("nonexistent", "report.pdf")


def test_create_storage_local():
    """create_storage with backend='local' returns LocalReportStorage."""
    config = {'report_storage_backend': 'local', 'report_storage_local_dir': '/tmp/reports'}
    storage = create_storage(config)
    assert isinstance(storage, LocalReportStorage)
