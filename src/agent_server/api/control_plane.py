"""Control Plane API — monitoring dashboard endpoints"""

from datetime import UTC, datetime, timedelta

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.orm import Assistant as AssistantORM
from ..core.orm import Run as RunORM
from ..core.orm import WorkerHeartbeat, get_session
from ..models.control_plane import (
    ActiveRun,
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
    active = await _get_active_runs(session)
    stats = await _get_stats(session)
    return ControlPlaneOverview(workers=workers, active_runs=active, stats=stats)


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
