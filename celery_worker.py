"""Celery worker entry point for long-running patent analysis tasks."""

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
    """Batch patent analysis -- skeleton, filled in Task 13."""
    from sources.long_task.status_manager import update_task_status, set_task_completed
    from sources.long_task.config import get_long_task_config

    ltc = get_long_task_config()
    max_patents = ltc['max_patents']
    patent_ids = list(dict.fromkeys(params.get('patent_ids', [])))[:max_patents]

    try:
        update_task_status(
            task_id,
            'generating_columns',
            0,
            f'正在启动分析任务（最多 {max_patents} 个专利）...',
        )
        # Placeholder: real logic in Task 13
        update_task_status(task_id, 'generating_columns', 5, '分析框架已生成')
        set_task_completed(task_id, [])
        return {'status': 'completed', 'task_id': task_id}
    except Exception as e:
        from sources.long_task.status_manager import set_task_failed

        set_task_failed(task_id, str(e))
        raise self.retry(exc=e)
