"""Celery tasks for sweep/cleanup operations and workflow execution."""

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import structlog
from croniter import croniter as croniter_cls
from graphs.workflow_engine.compiler import compile_workflow
from graphs.workflow_engine.nodes.base import fetch_ingestion_configurable
from graphs.workflow_engine.schema import WorkflowDefinition
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from .celery_app import celery_app
from .core.database import db_manager
from .core.orm import Run as RunORM
from .core.orm import (
    RunStatusHistory,
    Tenant,
    WorkerHeartbeat,
    Workflow,
    WorkflowRun,
    WorkflowSchedule,
    get_public_sync_session,
    new_tenant_sync_session,
)
from .services.internal_auth import create_internal_token

logger = structlog.getLogger(__name__)

HEARTBEAT_TIMEOUT_SECONDS = 45
ZOMBIE_RUN_TIMEOUT_MINUTES = 30
CANCELLATION_POLL_INTERVAL = 10  # seconds between DB polls for cancellation


async def _is_workflow_cancelled(async_engine, workflow_run_id: str, schema: str) -> bool:
    """Check if a workflow run has been cancelled. Bound to tenant schema."""
    engine = async_engine.execution_options(schema_translate_map={None: schema})
    async with AsyncSession(engine) as session:
        status = (
            await session.execute(select(WorkflowRun.status).where(WorkflowRun.id == workflow_run_id))
        ).scalar_one_or_none()
        return status == "cancelled"


async def _run_with_cancellation(coro, workflow_run_id: str, async_engine, schema: str):
    """Run a coroutine with cancellation support via DB polling.

    Creates an asyncio task for the main coroutine and a watcher task that
    polls the database every 10 seconds to check if the run has been cancelled.
    If cancelled, the main task is cancelled via asyncio.

    Returns the result of the coroutine, or None if cancelled.
    """
    # If already cancelled before we start, bail out
    if await _is_workflow_cancelled(async_engine, workflow_run_id, schema):
        logger.info("Workflow run already cancelled, skipping", run_id=workflow_run_id)
        coro.close()
        return None

    async def _cancellation_watcher(main_task: asyncio.Task):
        """Poll DB for cancellation and cancel the main task if needed."""
        try:
            while not main_task.done():
                await asyncio.sleep(CANCELLATION_POLL_INTERVAL)
                if await _is_workflow_cancelled(async_engine, workflow_run_id, schema):
                    logger.info(
                        "Workflow run cancelled, stopping task",
                        run_id=workflow_run_id,
                    )
                    main_task.cancel()
                    return
        except asyncio.CancelledError:
            pass

    main_task = asyncio.ensure_future(coro)
    watcher_task = asyncio.ensure_future(_cancellation_watcher(main_task))

    try:
        done, _ = await asyncio.wait(
            [main_task, watcher_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if main_task in done:
            watcher_task.cancel()
            try:
                return main_task.result()
            except asyncio.CancelledError:
                logger.info("Workflow run cancelled successfully", run_id=workflow_run_id)
                return None

        # Watcher finished first — main_task was already cancelled, wait for it
        try:
            await main_task
        except asyncio.CancelledError:
            logger.info("Workflow run cancelled successfully", run_id=workflow_run_id)
            return None

    finally:
        if not watcher_task.done():
            watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher_task


def _get_sync_engine():
    """Get a shared sync SQLAlchemy engine for Celery tasks."""
    db_manager.ensure_sync_engine()
    return db_manager.get_sync_engine()


async def _get_async_engine():
    """Get a shared async SQLAlchemy engine for Celery tasks."""
    try:
        return db_manager.get_engine()
    except RuntimeError:
        await db_manager.initialize()
        return db_manager.get_engine()


@celery_app.task(name="src.agent_server.tasks.sweep_stale_workers")
def sweep_stale_workers():
    """Mark workers as offline if their heartbeat is stale (>45s)."""
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
    """Mark runs stuck in active status for 30+ minutes as failed.

    ``runs`` lives in each tenant's schema, so we iterate over enabled
    tenants rather than relying on the default connection schema.
    """
    cutoff = datetime.now(UTC) - timedelta(minutes=ZOMBIE_RUN_TIMEOUT_MINUTES)
    error_msg = "Zombie run detected: no activity for 30+ minutes"
    total_marked = 0

    with get_public_sync_session() as public:
        tenants = public.execute(select(Tenant).where(Tenant.enabled.is_(True))).scalars().all()

    for tenant in tenants:
        with new_tenant_sync_session(tenant.schema) as session:
            stmt = select(RunORM).where(
                RunORM.status.in_(["pending", "running", "streaming"]),
                RunORM.updated_at < cutoff,
            )
            affected_runs = session.scalars(stmt).all()

            if not affected_runs:
                continue

            for run in affected_runs:
                history = RunStatusHistory(
                    run_id=run.run_id,
                    from_status=run.status,
                    to_status="failed",
                    error_message=error_msg,
                    created_at=datetime.now(UTC),
                )
                session.add(history)

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
            total_marked += result.rowcount
            logger.warning(
                "sweep_zombie_runs",
                tenant_uuid=tenant.uuid,
                marked_failed=result.rowcount,
                run_ids=zombie_run_ids,
            )

    return {"marked_failed": total_marked}


@celery_app.task(name="src.agent_server.tasks.cleanup_offline_workers")
def cleanup_offline_workers(max_age_hours: int = 0):
    """Delete offline workers. Use max_age_hours to keep recent ones (0 = all)."""
    engine = _get_sync_engine()

    with Session(engine) as session:
        conditions = [WorkerHeartbeat.status == "offline"]
        if max_age_hours > 0:
            cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
            conditions.append(WorkerHeartbeat.last_heartbeat < cutoff)
        result = session.execute(delete(WorkerHeartbeat).where(*conditions))
        session.commit()
        removed = result.rowcount

    if removed:
        logger.info("cleanup_offline_workers", removed=removed)
    return {"removed": removed}


@celery_app.task(
    name="src.agent_server.tasks.execute_workflow",
    bind=True,
    max_retries=0,
)
def execute_workflow(self, workflow_run_id: str, tenant_uuid: str, auth_token: str = ""):  # nosec B107
    """Execute a workflow run: compile JSON → LangGraph → ainvoke.

    Steps:
    0. Resolve tenant schema from ``tenant_uuid`` (public session)
    1. Load WorkflowRun from tenant schema, set status=running
    2. Load Workflow definition
    3. Validate and compile to StateGraph
    4. Execute via ainvoke (no streaming, no checkpointer)
    5. Save output or error to WorkflowRun
    """
    with get_public_sync_session() as public:
        row = public.execute(select(Tenant.schema, Tenant.enabled).where(Tenant.uuid == tenant_uuid)).one_or_none()

    if row is None or not row.enabled:
        logger.error(
            "execute_workflow: unknown or disabled tenant",
            tenant_uuid=tenant_uuid,
            run_id=workflow_run_id,
        )
        return {"error": "unknown tenant", "run_id": workflow_run_id}

    tenant_schema = row.schema

    with new_tenant_sync_session(tenant_schema) as session:
        run = session.execute(select(WorkflowRun).where(WorkflowRun.id == workflow_run_id)).scalar_one_or_none()

        if not run:
            logger.error("execute_workflow: run not found", run_id=workflow_run_id)
            return {"error": "run not found"}

        if run.status == "cancelled":
            logger.info("execute_workflow: run already cancelled", run_id=workflow_run_id)
            return {"status": "cancelled"}

        run.status = "running"
        run.started_at = datetime.now(UTC)
        session.commit()

        try:
            workflow = session.execute(select(Workflow).where(Workflow.id == run.workflow_id)).scalar_one_or_none()

            if not workflow:
                raise ValueError(f"Workflow {run.workflow_id} not found")

            definition = WorkflowDefinition(**workflow.definition)
            graph = compile_workflow(definition)
            compiled = graph.compile()

            initial_state = {
                "messages": [],
                "data": run.input_data or {},
                "steps": [],
            }

            async def _run():
                async_engine = await _get_async_engine()
                try:
                    ingestion_cfg = await fetch_ingestion_configurable(auth_token)
                    configurable = {
                        "auth_token": auth_token,
                        "team_id": run.team_id,
                        "user_id": run.user_id,
                        **ingestion_cfg,
                    }
                    return await _run_with_cancellation(
                        compiled.ainvoke(
                            initial_state,
                            config={"configurable": configurable},
                        ),
                        workflow_run_id,
                        async_engine,
                        tenant_schema,
                    )
                finally:
                    await async_engine.dispose()

            result = asyncio.run(_run())

            # Cancelled mid-execution
            if result is None:
                session.refresh(run)
                if run.status != "cancelled":
                    run.status = "cancelled"
                run.completed_at = datetime.now(UTC)
                session.commit()
                logger.info(
                    "execute_workflow: cancelled during execution",
                    run_id=workflow_run_id,
                )
                return {"status": "cancelled", "run_id": workflow_run_id}

            final_data = result.get("data", {})
            steps = result.get("steps", [])

            run.output_data = {**final_data, "steps": steps}
            run.status = "completed"
            run.completed_at = datetime.now(UTC)
            session.commit()

            logger.info(
                "execute_workflow: completed",
                run_id=workflow_run_id,
                workflow_name=definition.name,
            )
            return {"status": "completed", "run_id": workflow_run_id}

        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)[:2000]
            run.completed_at = datetime.now(UTC)
            session.commit()

            logger.error(
                "execute_workflow: failed",
                run_id=workflow_run_id,
                error=str(exc),
            )
            return {"status": "failed", "error": str(exc)}


@celery_app.task(name="src.agent_server.tasks.dispatch_scheduled_workflows")
def dispatch_scheduled_workflows():
    """Check for due workflow schedules and dispatch them.

    Iterates over every enabled tenant and queries schedules within that
    tenant's schema — schedules are stored per-tenant, so querying the
    public schema would miss them.
    """
    now = datetime.now(UTC)
    dispatched = 0

    with get_public_sync_session() as public:
        tenants = public.execute(select(Tenant).where(Tenant.enabled.is_(True))).scalars().all()

    for tenant in tenants:
        with new_tenant_sync_session(tenant.schema) as session:
            due_schedules = (
                session.execute(
                    select(WorkflowSchedule)
                    .where(
                        WorkflowSchedule.is_enabled.is_(True),
                        WorkflowSchedule.next_run_at <= now,
                    )
                    .with_for_update(skip_locked=True)
                )
                .scalars()
                .all()
            )

            for schedule in due_schedules:
                workflow = session.execute(
                    select(Workflow).where(
                        Workflow.id == schedule.workflow_id,
                        Workflow.is_active.is_(True),
                        Workflow.deleted_at.is_(None),
                    )
                ).scalar_one_or_none()

                if not workflow:
                    schedule.is_enabled = False
                    schedule.updated_at = now
                    continue

                # Generate internal JWT so downstream nodes can call backend
                # on behalf of the workflow owner (aud = this tenant's uuid).
                auth_token = ""  # nosec B105
                token = create_internal_token(
                    workflow.user_id,
                    workflow.team_id,
                    tenant.uuid,
                )
                if token:
                    auth_token = f"Bearer {token}"

                run = WorkflowRun(
                    workflow_id=schedule.workflow_id,
                    team_id=schedule.team_id,
                    user_id=schedule.user_id,
                    status="pending",
                    input_data=schedule.input_data or {},
                )
                session.add(run)
                session.flush()

                task = execute_workflow.delay(run.id, tenant_uuid=tenant.uuid, auth_token=auth_token)
                run.celery_task_id = task.id

                schedule.last_run_at = now
                schedule.updated_at = now
                tz = ZoneInfo(schedule.timezone)
                cron = croniter_cls(schedule.cron_expression, now.astimezone(tz))
                schedule.next_run_at = cron.get_next(datetime).astimezone(UTC)

                dispatched += 1

            session.commit()

    if dispatched:
        logger.info("dispatch_scheduled_workflows", dispatched=dispatched)
    return {"dispatched": dispatched}
