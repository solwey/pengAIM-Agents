"""Celery application for pengAIM-agents background tasks."""

import os

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "2")
REDIS_USER = os.getenv("REDIS_USER", "")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_PROTOCOL = os.getenv("REDIS_PROTOCOL", "redis")
REDIS_USE_CREDENTIALS = os.getenv("REDIS_USE_CREDENTIALS", "").strip().lower() in {
    "true",
    "1",
    "yes",
    "y",
    "on",
}

if REDIS_USE_CREDENTIALS:
    auth_part = f"{REDIS_USER}:{REDIS_PASSWORD}@"
else:
    auth_part = ""

REDIS_URL = f"{REDIS_PROTOCOL}://{auth_part}{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
if REDIS_PROTOCOL == "rediss":
    REDIS_URL += "?ssl_cert_reqs=CERT_OPTIONAL"

celery_app = Celery(
    "agent_server",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.agent_server.tasks"],
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
