"""Control Plane API — monitoring dashboard endpoints"""

from datetime import UTC, datetime, timedelta

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.orm import Assistant as AssistantORM
from ..core.orm import Run as RunORM
from ..core.orm import WorkerHeartbeat, get_session
from ..models.control_plane import (
    ActiveRun,
    CeleryWorkerStatus,
    ControlPlaneOverview,
    DashboardStats,
    WorkerStatus,
)

logger = structlog.getLogger(__name__)
router = APIRouter(prefix="/control-plane", tags=["Control Plane"])

HEARTBEAT_TIMEOUT_SECONDS = 45


def _worker_effective_status(hb: WorkerHeartbeat) -> str:
    if hb.status == "offline":
        return "offline"
    age = (datetime.now(UTC) - hb.last_heartbeat).total_seconds()
    return "offline" if age > HEARTBEAT_TIMEOUT_SECONDS else "online"


# ---------- Overview ----------


@router.get("/overview", response_model=ControlPlaneOverview)
async def get_overview(session: AsyncSession = Depends(get_session)):
    """Combined dashboard snapshot: workers + active runs + 24h stats."""
    workers = await _get_workers(session)
    celery_workers = await _get_celery_workers()
    active = await _get_active_runs(session)
    stats = await _get_stats(session)
    return ControlPlaneOverview(
        workers=workers,
        celery_workers=celery_workers,
        active_runs=active,
        stats=stats,
    )


# ---------- Workers ----------


@router.get("/workers", response_model=list[WorkerStatus])
async def list_workers(session: AsyncSession = Depends(get_session)):
    return await _get_workers(session)


async def _get_workers(session: AsyncSession) -> list[WorkerStatus]:
    cutoff_24h = datetime.now(UTC) - timedelta(hours=24)
    result = await session.scalars(
        select(WorkerHeartbeat).where(WorkerHeartbeat.last_heartbeat > cutoff_24h)
    )
    now = datetime.now(UTC)
    workers = []
    for hb in result:
        uptime = (now - hb.started_at).total_seconds()
        workers.append(
            WorkerStatus(
                id=hb.id,
                status=_worker_effective_status(hb),
                started_at=hb.started_at,
                last_heartbeat=hb.last_heartbeat,
                uptime_seconds=uptime,
                active_run_count=hb.active_run_count,
                metadata=hb.metadata_dict,
            )
        )
    return workers


@router.delete("/workers/cleanup")
async def cleanup_offline_workers(session: AsyncSession = Depends(get_session)):
    """Delete all offline worker heartbeat records."""
    result = await session.execute(
        delete(WorkerHeartbeat).where(WorkerHeartbeat.status == "offline")
    )
    await session.commit()
    removed = result.rowcount
    if removed:
        logger.info("cleanup_offline_workers", removed=removed)
    return {"removed": removed}


# ---------- Celery Workers ----------


@router.get("/celery-workers", response_model=list[CeleryWorkerStatus])
async def list_celery_workers():
    """List Celery workers and their status."""
    return await _get_celery_workers()


async def _get_celery_workers() -> list[CeleryWorkerStatus]:
    """Query Celery worker stats via inspect API."""
    import asyncio

    from ..celery_app import celery_app

    def _inspect():
        inspector = celery_app.control.inspect(timeout=1.0)
        try:
            active = inspector.active() or {}
            registered = inspector.registered() or {}
            stats = inspector.stats() or {}
        except Exception:
            return {}, {}, {}
        return active, registered, stats

    try:
        active, registered, stats = await asyncio.to_thread(_inspect)
    except Exception:
        logger.warning("Failed to inspect Celery workers")
        return []

    workers = []
    all_names = set(active.keys()) | set(registered.keys()) | set(stats.keys())
    for name in sorted(all_names):
        worker_stats = stats.get(name, {})
        pool = worker_stats.get("pool", {})
        rusage = worker_stats.get("rusage", {})
        broker = worker_stats.get("broker", {})
        total_tasks = worker_stats.get("total", {})
        workers.append(
            CeleryWorkerStatus(
                name=name,
                status="online",
                active_tasks=len(active.get(name, [])),
                registered_tasks=registered.get(name, []),
                pool_size=pool.get("max-concurrency"),
                pid=worker_stats.get("pid"),
                metadata={
                    "total_tasks": total_tasks,
                    "total_executed": sum(total_tasks.values()) if total_tasks else 0,
                    "broker_transport": broker.get("transport"),
                    "broker_host": broker.get("hostname"),
                    "broker_port": broker.get("port"),
                    "prefetch_count": worker_stats.get("prefetch_count"),
                    "memory_mb": round(rusage.get("maxrss", 0) / 1024 / 1024, 1) if rusage.get("maxrss") else None,
                    "cpu_user_time": round(rusage.get("utime", 0), 2) if rusage.get("utime") else None,
                    "cpu_system_time": round(rusage.get("stime", 0), 2) if rusage.get("stime") else None,
                    "pool_processes": len(pool.get("processes", [])),
                    "pool_implementation": pool.get("implementation", "").split(":")[-1] if pool.get("implementation") else None,
                },
            )
        )
    return workers


# ---------- Active Runs ----------


async def _get_active_runs(session: AsyncSession) -> list[ActiveRun]:
    now = datetime.now(UTC)
    stmt = (
        select(RunORM, AssistantORM.name)
        .outerjoin(AssistantORM, RunORM.assistant_id == AssistantORM.assistant_id)
        .where(RunORM.status.in_(["pending", "running", "streaming"]))
        .order_by(RunORM.created_at.desc())
    )
    rows = (await session.execute(stmt)).all()
    return [
        ActiveRun(
            run_id=run.run_id,
            thread_id=run.thread_id,
            assistant_id=run.assistant_id,
            assistant_name=name,
            status=run.status,
            current_step=run.current_step,
            duration_seconds=(now - run.created_at).total_seconds(),
            created_at=run.created_at,
            user_id=run.user_id,
        )
        for run, name in rows
    ]


# ---------- 24h Stats ----------


async def _get_stats(session: AsyncSession) -> DashboardStats:
    since = datetime.now(UTC) - timedelta(hours=24)

    total = await session.scalar(
        select(func.count()).select_from(RunORM).where(RunORM.created_at >= since)
    ) or 0

    completed = await session.scalar(
        select(func.count())
        .select_from(RunORM)
        .where(RunORM.created_at >= since, RunORM.status == "completed")
    ) or 0

    failed = await session.scalar(
        select(func.count())
        .select_from(RunORM)
        .where(RunORM.created_at >= since, RunORM.status == "failed")
    ) or 0

    avg_dur = await session.scalar(
        select(func.avg(RunORM.duration_ms))
        .where(
            RunORM.created_at >= since,
            RunORM.duration_ms.isnot(None),
        )
    )

    return DashboardStats(
        total_runs_24h=total,
        completed_24h=completed,
        failed_24h=failed,
        avg_duration_ms=float(avg_dur) if avg_dur else None,
    )
