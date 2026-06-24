"""Tencent COS (Cloud Object Storage) report storage backend."""
import asyncio
import os
from io import BytesIO
from sources.long_task.storage import ReportStorage


class COSReportStorage(ReportStorage):
    """COS-backed report storage.

    Reports are stored under: {prefix}/{task_id}/{filename}
    """

    def __init__(
        self,
        bucket: str,
        region: str = "ap-hongkong",
        secret_id: str = "",
        secret_key: str = "",
        prefix: str = "reports",
    ):
        self.bucket = bucket
        self.region = region
        self.prefix = prefix
        self._secret_id = secret_id
        self._secret_key = secret_key
        self._client = None

    @property
    def client(self):
        """Lazy-init the COS client (not thread-safe, fine for single-worker Celery)."""
        if self._client is None:
            from qcloud_cos import CosConfig, CosS3Client
            config = CosConfig(
                Region=self.region,
                SecretId=self._secret_id,
                SecretKey=self._secret_key,
            )
            self._client = CosS3Client(config)
        return self._client

    def _object_key(self, task_id: str, filename: str) -> str:
        return f"{self.prefix}/{task_id}/{filename}"

    def _task_prefix(self, task_id: str) -> str:
        return f"{self.prefix}/{task_id}/"

    def _put_sync(self, task_id: str, filename: str, content: bytes) -> str:
        key = self._object_key(task_id, filename)
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=BytesIO(content),
            EnableMD5=False,
        )
        return key

    def _get_sync(self, task_id: str, filename: str) -> bytes:
        key = self._object_key(task_id, filename)
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        # response['Body'] is a stream; read into bytes
        return response["Body"].get_raw_stream().read()

    def _delete_sync(self, task_id: str) -> None:
        prefix = self._task_prefix(task_id)
        # List all objects under the task prefix
        marker = ""
        while True:
            resp = self.client.list_objects(
                Bucket=self.bucket,
                Prefix=prefix,
                Marker=marker,
                MaxKeys=100,
            )
            contents = resp.get("Contents") or []
            if not contents:
                break
            objects = [{"Key": obj["Key"]} for obj in contents]
            self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Object": objects, "Quiet": "true"},
            )
            if resp.get("IsTruncated") == "false":
                break
            marker = resp.get("NextMarker", contents[-1]["Key"])

    async def put(self, task_id: str, filename: str, content: bytes) -> str:
        return await asyncio.to_thread(self._put_sync, task_id, filename, content)

    async def get(self, task_id: str, filename: str) -> bytes:
        try:
            return await asyncio.to_thread(self._get_sync, task_id, filename)
        except Exception as e:
            error_msg = str(e)
            if "NoSuchKey" in error_msg or "not found" in error_msg.lower():
                raise FileNotFoundError(f"Report not found in COS: {self._object_key(task_id, filename)}")
            raise

    async def delete(self, task_id: str) -> None:
        await asyncio.to_thread(self._delete_sync, task_id)
