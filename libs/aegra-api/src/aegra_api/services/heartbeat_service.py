"""Heartbeat service for worker health monitoring"""

import asyncio
import contextlib
import os
import platform
import resource
import sys
from datetime import UTC, datetime

import structlog
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ..core.orm import WorkerHeartbeat, _get_session_maker

logger = structlog.getLogger(__name__)

HEARTBEAT_INTERVAL = 15  # seconds
HEARTBEAT_TIMEOUT = 45  # seconds — mark offline after this


class HeartbeatService:
    def __init__(self, active_runs: dict):
        self._active_runs = active_runs
        self._task: asyncio.Task | None = None
        self._started_at = datetime.now(UTC)
        self._worker_id = f"{platform.node()}_{os.getpid()}"

    @property
    def worker_id(self) -> str:
        return self._worker_id

    async def start(self) -> None:
        """Register worker and start heartbeat loop."""
        await self._upsert_heartbeat()
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Heartbeat service started", worker_id=self._worker_id)

    async def stop(self) -> None:
        """Mark worker offline and stop loop."""
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

        try:
            maker = _get_session_maker()
            async with maker() as session:
                await session.execute(
                    update(WorkerHeartbeat)
                    .where(WorkerHeartbeat.id == self._worker_id)
                    .values(status="offline", last_heartbeat=datetime.now(UTC))
                )
                await session.commit()
        except Exception:
            logger.exception("Failed to mark worker offline", worker_id=self._worker_id)

        logger.info("Heartbeat service stopped", worker_id=self._worker_id)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                await self._upsert_heartbeat()
            except Exception:
                logger.exception("Heartbeat update failed", worker_id=self._worker_id)

    async def _upsert_heartbeat(self) -> None:
        now = datetime.now(UTC)
        active_count = sum(1 for t in self._active_runs.values() if not t.done())
        # RSS in MB — ru_maxrss is bytes on macOS, KB on Linux
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = usage.ru_maxrss / (1024 * 1024) if sys.platform == "darwin" else usage.ru_maxrss / 1024
        metadata = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
            "memory_rss_mb": round(rss_mb, 1),
            "total_runs_processed": sum(1 for t in self._active_runs.values() if t.done()),
        }

        maker = _get_session_maker()
        async with maker() as session:
            stmt = pg_insert(WorkerHeartbeat.__table__).values(
                id=self._worker_id,
                status="online",
                started_at=self._started_at,
                last_heartbeat=now,
                active_run_count=active_count,
                metadata=metadata,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "status": "online",
                    "last_heartbeat": now,
                    "active_run_count": active_count,
                    "metadata": metadata,
                },
            )
            await session.execute(stmt)
            await session.commit()


# Singleton instance — initialized in main.py lifespan
heartbeat_service: HeartbeatService | None = None
