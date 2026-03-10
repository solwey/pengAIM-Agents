"""Celery application for pengAIM-agents background tasks."""

import os

from celery import Celery
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "2")
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

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
    },
)
