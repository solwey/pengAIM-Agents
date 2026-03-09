"""Control Plane API — monitoring dashboard endpoints"""

from datetime import UTC, datetime, timedelta

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .runs import active_runs 

from ..core.orm import Assistant as AssistantORM
from ..core.orm import Run as RunORM
from ..core.orm import RunStatusHistory, WorkerHeartbeat, get_session
from ..models.control_plane import (
    ActiveRun,
    ControlPlaneOverview,
    DashboardStats,
    RunHistoryEntry,
    RunHistoryPage,
    RunStatusTransition,
    WorkerStatus,
)
from ..services.streaming_service import streaming_service

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


@router.post("/workers/sweep")
async def sweep_stale_workers(session: AsyncSession = Depends(get_session)):
    """Mark workers as offline if their heartbeat is stale (>45s)."""
    cutoff = datetime.now(UTC) - timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS)
    result = await session.execute(
        update(WorkerHeartbeat)
        .where(
            WorkerHeartbeat.last_heartbeat < cutoff,
            WorkerHeartbeat.status != "offline",
        )
        .values(status="offline")
    )
    await session.commit()
    marked = result.rowcount
    logger.info("sweep_stale_workers", marked_offline=marked)
    return {"marked_offline": marked}


@router.delete("/workers/cleanup")
async def cleanup_offline_workers(
    max_age_hours: int = Query(0, description="Only delete offline workers older than N hours (0 = all)"),
    session: AsyncSession = Depends(get_session),
):
    """Delete offline workers. Use max_age_hours to keep recent ones."""
    conditions = [WorkerHeartbeat.status == "offline"]
    if max_age_hours > 0:
        cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
        conditions.append(WorkerHeartbeat.last_heartbeat < cutoff)
    result = await session.execute(
        delete(WorkerHeartbeat).where(*conditions)
    )
    await session.commit()
    removed = result.rowcount
    logger.info("cleanup_offline_workers", removed=removed)
    return {"removed": removed}


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


# ---------- Run History ----------


SORT_COLUMNS = {
    "created_at": RunORM.created_at,
    "status": RunORM.status,
    "duration_ms": RunORM.duration_ms,
    "updated_at": RunORM.updated_at,
}


@router.get("/runs", response_model=RunHistoryPage)
async def list_runs(
    status: str | None = Query(None),
    since: str | None = Query(None, description="ISO datetime"),
    assistant_id: str | None = Query(None, description="Assistant ID or name"),
    search: str | None = Query(None, description="Search by run_id prefix"),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """Paginated run history with optional status/date/assistant/search filters."""
    base = select(RunORM, AssistantORM.name).outerjoin(
        AssistantORM, RunORM.assistant_id == AssistantORM.assistant_id
    )
    count_base = select(func.count()).select_from(RunORM).outerjoin(
        AssistantORM, RunORM.assistant_id == AssistantORM.assistant_id
    )

    if status:
        statuses = [s.strip() for s in status.split(",")]
        base = base.where(RunORM.status.in_(statuses))
        count_base = count_base.where(RunORM.status.in_(statuses))

    if since:
        since_dt = datetime.fromisoformat(since)
        base = base.where(RunORM.created_at >= since_dt)
        count_base = count_base.where(RunORM.created_at >= since_dt)

    if assistant_id:
        # Support filtering by name (from deduplicated dropdown) or by ID
        assistant_filter = or_(
            RunORM.assistant_id == assistant_id,
            AssistantORM.name == assistant_id,
        )
        base = base.where(assistant_filter)
        count_base = count_base.where(assistant_filter)

    if search:
        pattern = f"%{search}%"
        search_filter = or_(
            RunORM.run_id.ilike(f"{search}%"),
            AssistantORM.name.ilike(pattern),
        )
        base = base.where(search_filter)
        count_base = count_base.where(search_filter)

    total = await session.scalar(count_base) or 0

    sort_col = SORT_COLUMNS.get(sort_by, RunORM.created_at)
    order = sort_col.asc() if sort_order == "asc" else sort_col.desc()
    stmt = base.order_by(order).limit(limit).offset(offset)
    rows = (await session.execute(stmt)).all()

    runs = [
        RunHistoryEntry(
            run_id=run.run_id,
            thread_id=run.thread_id,
            assistant_id=run.assistant_id,
            assistant_name=name,
            status=run.status,
            error_message=run.error_message,
            duration_ms=run.duration_ms,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )
        for run, name in rows
    ]
    return RunHistoryPage(runs=runs, total=total, limit=limit, offset=offset)


# ---------- Assistants List (for filters) ----------


@router.get("/assistants")
async def list_assistants_for_filter(session: AsyncSession = Depends(get_session)):
    """List unique assistant names for filter dropdowns."""
    result = await session.execute(
        select(AssistantORM.name)
        .where(AssistantORM.deleted_at.is_(None))
        .where(AssistantORM.name.isnot(None))
        .group_by(AssistantORM.name)
        .order_by(AssistantORM.name)
    )
    return [{"id": name, "name": name} for (name,) in result.all()]


# ---------- Run Status History ----------


@router.get("/runs/{run_id}/history", response_model=list[RunStatusTransition])
async def get_run_history(
    run_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Status transition timeline for a specific run."""
    result = await session.scalars(
        select(RunStatusHistory)
        .where(RunStatusHistory.run_id == run_id)
        .order_by(RunStatusHistory.created_at.asc())
    )
    return [
        RunStatusTransition(
            from_status=h.from_status,
            to_status=h.to_status,
            error_message=h.error_message,
            traceback=h.traceback,
            created_at=h.created_at,
        )
        for h in result
    ]


# ---------- Cancel Run ----------


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Cancel a running task."""
    logger.info("cancel_run_requested", run_id=run_id)
    run = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
    if not run:
        raise HTTPException(404, f"Run '{run_id}' not found")
    if run.status not in ("pending", "running", "streaming"):
        raise HTTPException(409, f"Run is not active (status={run.status})")

    # Record status transition
    history = RunStatusHistory(
        run_id=run_id,
        from_status=run.status,
        to_status="cancelled",
        created_at=datetime.now(UTC),
    )
    session.add(history)

    # Update run status in DB
    await session.execute(
        update(RunORM)
        .where(RunORM.run_id == run_id)
        .values(status="cancelled", error_message="Cancelled via control plane")
    )
    await session.commit()

    # Cancel streaming and asyncio task
    await streaming_service.cancel_run(run_id)

    task = active_runs.get(run_id)
    if task and not task.done():
        task.cancel()

    logger.info("cancel_run_completed", run_id=run_id)
    return {"status": "cancelled", "run_id": run_id}


# ---------- Sweep Zombie Runs ----------

ZOMBIE_RUN_TIMEOUT_MINUTES = 30


@router.post("/runs/sweep")
async def sweep_zombie_runs(session: AsyncSession = Depends(get_session)):
    """Mark runs stuck in active status for 30+ minutes as failed."""
    cutoff = datetime.now(UTC) - timedelta(minutes=ZOMBIE_RUN_TIMEOUT_MINUTES)
    error_msg = "Zombie run detected: no activity for 30+ minutes"

    # Find affected runs before updating
    stmt = select(RunORM).where(
        RunORM.status.in_(["pending", "running", "streaming"]),
        RunORM.updated_at < cutoff,
    )
    affected_runs = (await session.scalars(stmt)).all()

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
    result = await session.execute(update_stmt)
    await session.commit()

    zombie_run_ids = [r.run_id for r in affected_runs]
    marked = result.rowcount
    logger.warning("sweep_zombie_runs", marked_failed=marked, run_ids=zombie_run_ids)
    return {"marked_failed": marked}
