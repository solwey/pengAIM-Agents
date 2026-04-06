"""Celery application for pengAIM-agents background tasks."""

from celery import Celery

from aegra_api.settings import settings

celery_app = Celery(
    "agent_server",
    broker=settings.redis.redis_url,
    backend=settings.redis.redis_url,
    include=["aegra_api.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "sweep-stale-workers-every-30s": {
            "task": "src.agent_server.tasks.sweep_stale_workers",
            "schedule": 30.0,
        },
        "sweep-zombie-runs-every-60s": {
            "task": "src.agent_server.tasks.sweep_zombie_runs",
            "schedule": 60.0,
        },
        "cleanup-offline-workers-every-5m": {
            "task": "src.agent_server.tasks.cleanup_offline_workers",
            "schedule": 300.0,
        },
        "dispatch-scheduled-workflows-every-60s": {
            "task": "src.agent_server.tasks.dispatch_scheduled_workflows",
            "schedule": 60.0,
        },
    },
)
