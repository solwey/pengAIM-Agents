"""Celery tasks for sweep/cleanup operations."""

import os
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session

from .celery_app import celery_app

logger = structlog.getLogger(__name__)

HEARTBEAT_TIMEOUT_SECONDS = 45
ZOMBIE_RUN_TIMEOUT_MINUTES = 30


def _get_sync_engine():
    """Create a sync SQLAlchemy engine for Celery tasks."""
    url = os.getenv(
        "DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/aegra"
    )
    # Convert async URL to sync
    sync_url = url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
    return create_engine(sync_url)


@celery_app.task(name="src.agent_server.tasks.sweep_stale_workers")
def sweep_stale_workers():
    """Mark workers as offline if their heartbeat is stale (>45s)."""
    from .core.orm import WorkerHeartbeat

    engine = _get_sync_engine()
    cutoff = datetime.now(UTC) - timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS)

    with Session(engine) as session:
        result = session.execute(
            update(WorkerHeartbeat)
            .where(
                WorkerHeartbeat.last_heartbeat < cutoff,
                WorkerHeartbeat.status != "offline",
            )
            .values(status="offline")
        )
        session.commit()
        marked = result.rowcount

    if marked:
        logger.info("sweep_stale_workers", marked_offline=marked)
    return {"marked_offline": marked}


@celery_app.task(name="src.agent_server.tasks.sweep_zombie_runs")
def sweep_zombie_runs():
    """Mark runs stuck in active status for 30+ minutes as failed."""
    from .core.orm import Run as RunORM
    from .core.orm import RunStatusHistory

    engine = _get_sync_engine()
    cutoff = datetime.now(UTC) - timedelta(minutes=ZOMBIE_RUN_TIMEOUT_MINUTES)
    error_msg = "Zombie run detected: no activity for 30+ minutes"

    with Session(engine) as session:
        from sqlalchemy import select

        # Find affected runs
        stmt = select(RunORM).where(
            RunORM.status.in_(["pending", "running", "streaming"]),
            RunORM.updated_at < cutoff,
        )
        affected_runs = session.scalars(stmt).all()

        if not affected_runs:
            return {"marked_failed": 0}

        # Insert status history for each affected run
        for run in affected_runs:
            history = RunStatusHistory(
                run_id=run.run_id,
                from_status=run.status,
                to_status="failed",
                error_message=error_msg,
                created_at=datetime.now(UTC),
            )
            session.add(history)

        # Bulk update runs to failed
        update_stmt = (
            update(RunORM)
            .where(
                RunORM.status.in_(["pending", "running", "streaming"]),
                RunORM.updated_at < cutoff,
            )
            .values(status="failed", error_message=error_msg)
        )
        result = session.execute(update_stmt)
        session.commit()

        zombie_run_ids = [r.run_id for r in affected_runs]
        marked = result.rowcount
        logger.warning(
            "sweep_zombie_runs", marked_failed=marked, run_ids=zombie_run_ids
        )
        return {"marked_failed": marked}
