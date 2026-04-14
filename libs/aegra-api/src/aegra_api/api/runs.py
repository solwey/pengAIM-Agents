"""Run endpoints for Agent Protocol"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.core.active_runs import active_runs
from aegra_api.core.auth_deps import auth_dependency, get_current_user
from aegra_api.core.auth_handlers import build_auth_context, handle_event
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import RunEvent as RunEventORM
from aegra_api.core.orm import RunStatusHistory, Tenant, get_session
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.sse import create_end_event, get_sse_headers
from aegra_api.core.tenant import get_current_tenant
from aegra_api.models import Run, RunCreate, RunStatus, User
from aegra_api.models.control_plane import (
    RunDetailResponse,
    RunHistoryEntry,
    RunHistoryPage,
    RunStatusTransition,
)
from aegra_api.models.errors import CONFLICT, NOT_FOUND, SSE_RESPONSE
from aegra_api.services.run_preparation import _prepare_run
from aegra_api.services.run_waiters import TERMINAL_STATES, encode_output, heartbeat_wait_body
from aegra_api.services.streaming_service import streaming_service
from aegra_api.settings import settings
from aegra_api.utils.status_compat import validate_run_status

router = APIRouter(tags=["Thread Runs"], dependencies=auth_dependency)

logger = structlog.getLogger(__name__)


# active_runs is imported from aegra_api.core.active_runs (dependency-free module)

# Default stream modes for background run execution
DEFAULT_STREAM_MODES = ["values"]


@router.post("/threads/{thread_id}/runs", response_model=Run, responses={**NOT_FOUND, **CONFLICT})
async def create_run(
    thread_id: str,
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
) -> Run:
    """Create and execute a new run.

    Starts graph execution asynchronously and returns the run record
    immediately with status `pending`. Poll the run or use the stream
    endpoint to follow progress. Provide either `input` or `command` (for
    human-in-the-loop resumption) but not both.
    """
    # Authorization check (create_run action on threads resource)
    ctx = build_auth_context(user, "threads", "create_run")
    value = {**request.model_dump(), "thread_id": thread_id}
    filters = await handle_event(ctx, value)

    # If handler modified config/context, update request
    if filters:
        if "config" in filters and isinstance(filters["config"], dict):
            request.config = {**(request.config or {}), **filters["config"]}
        if "context" in filters and isinstance(filters["context"], dict):
            request.context = {**(request.context or {}), **filters["context"]}
    else:
        value_config = value.get("config")
        if isinstance(value_config, dict):
            request.config = {**(request.config or {}), **value_config}

        value_context = value.get("context")
        if isinstance(value_context, dict):
            request.context = {**(request.context or {}), **value_context}

    _run_id, run, _job = await _prepare_run(session, thread_id, request, user, tenant, initial_status="pending")

    return run


@router.post("/threads/{thread_id}/runs/stream", responses={**SSE_RESPONSE, **NOT_FOUND, **CONFLICT})
async def create_and_stream_run(
    thread_id: str,
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
) -> StreamingResponse:
    """Create a new run and stream its execution via SSE.

    Returns a `text/event-stream` response with Server-Sent Events. Each
    event has a `type` field (e.g. `values`, `updates`, `messages`,
    `metadata`, `end`) and a JSON `data` payload.

    Set `on_disconnect` to `"continue"` if the run should keep executing
    after the client disconnects (default is `"cancel"`). Use `stream_mode`
    to control which event types are emitted.
    """
    run_id, run, _job = await _prepare_run(session, thread_id, request, user, tenant, initial_status="pending")

    # Default to cancel on disconnect - this matches user expectation that clicking
    # "Cancel" in the frontend will stop the backend task. Users can explicitly
    # set on_disconnect="continue" if they want the task to continue.
    cancel_on_disconnect = (request.on_disconnect or "cancel").lower() == "cancel"

    return StreamingResponse(
        streaming_service.stream_run_execution(
            tenant,
            run,
            None,
            cancel_on_disconnect=cancel_on_disconnect,
        ),
        media_type="text/event-stream",
        headers={
            **get_sse_headers(),
            "Location": f"/threads/{thread_id}/runs/{run_id}/stream",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )


@router.get("/threads/{thread_id}/runs/{run_id}", response_model=Run, responses={**NOT_FOUND})
async def get_run(
    thread_id: str,
    run_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Get a run by its ID.

    Returns the current state of the run including its status, input, output,
    and error information.
    """
    # Authorization check (read action on runs resource)
    ctx = build_auth_context(user, "runs", "read")
    value = {"run_id": run_id, "thread_id": thread_id}
    await handle_event(ctx, value)

    stmt = (
        select(RunORM)
        .join(ThreadORM, ThreadORM.thread_id == RunORM.thread_id)
        .where(
            RunORM.run_id == str(run_id),
            RunORM.thread_id == thread_id,
            RunORM.team_id == user.team_id,
            or_(
                RunORM.user_id == user.id,
                ThreadORM.is_shared.is_(True),
            ),
        )
    )
    logger.info(f"[get_run] querying DB run_id={run_id} thread_id={thread_id} user={user.id} team={user.team_id}")
    run_orm = await session.scalar(stmt)
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    # Refresh to ensure we have the latest data (in case background task updated it)
    await session.refresh(run_orm)

    logger.info(f"[get_run] found run status={run_orm.status} user={user.id} thread_id={thread_id} run_id={run_id}")
    # Convert to Pydantic
    # noinspection PyTypeChecker
    return Run.model_validate(run_orm)


@router.get("/threads/{thread_id}/runs", response_model=list[Run])
async def list_runs(
    thread_id: str,
    limit: int = Query(10, ge=1, description="Maximum number of runs to return"),
    offset: int = Query(0, ge=0, description="Number of runs to skip for pagination"),
    status: str | None = Query(
        None, description="Filter by run status (e.g. pending, running, success, error, interrupted)"
    ),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[Run]:
    """List runs for a thread.

    Returns runs ordered by creation time (newest first). Use `status` to
    filter and `limit`/`offset` to paginate.
    """
    stmt = (
        select(RunORM)
        .join(ThreadORM, ThreadORM.thread_id == RunORM.thread_id)
        .where(
            RunORM.thread_id == thread_id,
            RunORM.team_id == user.team_id,
        )
    )

    if status:
        stmt = stmt.where(RunORM.status == status)

    stmt = stmt.where(
        or_(
            RunORM.user_id == user.id,
            ThreadORM.is_shared.is_(True),
        )
    )

    stmt = stmt.order_by(RunORM.created_at.desc()).limit(limit).offset(offset)

    logger.info(f"[list_runs] querying DB thread_id={thread_id} user={user.id}")
    result = await session.scalars(stmt)
    rows = result.all()

    runs = [Run.model_validate(r) for r in rows]
    logger.info(f"[list_runs] total={len(runs)} user={user.id} thread_id={thread_id}")
    return runs


@router.patch("/threads/{thread_id}/runs/{run_id}", response_model=Run, responses={**NOT_FOUND})
async def update_run(
    thread_id: str,
    run_id: str,
    request: RunStatus,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Update a run's status.

    Primarily used to interrupt a running execution. Set `status` to
    `"interrupted"` to cooperatively stop the run.
    """
    logger.info(f"[update_run] fetch for update run_id={run_id} thread_id={thread_id} user={user.id}")
    stmt = (
        select(RunORM)
        .join(ThreadORM, ThreadORM.thread_id == RunORM.thread_id)
        .where(
            RunORM.run_id == str(run_id),
            RunORM.thread_id == thread_id,
            RunORM.team_id == user.team_id,
        )
    )

    stmt = stmt.where(
        or_(
            RunORM.user_id == user.id,
            ThreadORM.is_shared.is_(True),
        )
    )
    run_orm = await session.scalar(stmt)
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    # Handle interruption/cancellation
    # Validate status conforms to API specification
    validated_status = validate_run_status(request.status)

    if validated_status == "interrupted":
        logger.info(f"[update_run] cancelling/interrupting run_id={run_id} user={user.id} thread_id={thread_id}")
        # Handle interruption - use interrupt_run for cooperative interruption
        await streaming_service.interrupt_run(run_id)
        logger.info(f"[update_run] set DB status=interrupted run_id={run_id}")
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == str(run_id))
            .values(status="interrupted", updated_at=datetime.now(UTC))
        )
        await session.commit()
        logger.info(f"[update_run] commit done (interrupted) run_id={run_id}")

    # Return the final run state
    run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")
    # Refresh to ensure we have the latest data after our own update
    await session.refresh(run_orm)
    # noinspection PyTypeChecker
    return Run.model_validate(run_orm)


@router.get("/threads/{thread_id}/runs/{run_id}/join", responses={**NOT_FOUND})
async def join_run(
    thread_id: str,
    run_id: str,
    user: User = Depends(get_current_user),
    tenant: Tenant = Depends(get_current_tenant),
) -> StreamingResponse:
    """Wait for a run to complete and return its output.

    Returns a chunked ``application/json`` response. While the run is still
    executing, the server sends periodic ``\\n`` heartbeat bytes to keep the
    connection alive through proxies and load balancers (AWS ALB, Cloudflare,
    etc.). The final chunk is the JSON result. Leading whitespace is ignored
    by JSON parsers, so clients can parse the concatenated body normally.

    If the run is already in a terminal state, the output is returned
    immediately with no heartbeat overhead.

    Sessions are managed manually (not via ``Depends``) to avoid holding a
    pool connection during the long wait.
    """
    # Short-lived session: validate run exists and check terminal state
    async for session in get_session(tenant):
        stmt = (
            select(RunORM)
            .join(ThreadORM, ThreadORM.thread_id == RunORM.thread_id)
            .where(
                RunORM.run_id == str(run_id),
                RunORM.thread_id == thread_id,
                RunORM.team_id == user.team_id,
            )
        )

        stmt = stmt.where(
            or_(
                RunORM.user_id == user.id,
                ThreadORM.is_shared.is_(True),
            )
        )

        run_orm = await session.scalar(stmt)
        if not run_orm:
            raise HTTPException(404, f"Run '{run_id}' not found")

        if run_orm.status in TERMINAL_STATES:
            return StreamingResponse(
                iter([encode_output(run_orm.output or {})]),
                media_type="application/json",
            )

    return StreamingResponse(
        heartbeat_wait_body(
            run_id,
            thread_id,
            user.id,
            tenant_schema=tenant.schema,
            timeout=settings.worker.BG_JOB_TIMEOUT_SECS,
        ),
        media_type="application/json",
        headers={
            "Location": f"/threads/{thread_id}/runs/{run_id}/join",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )


@router.post("/threads/{thread_id}/runs/wait", responses={**NOT_FOUND, **CONFLICT})
async def wait_for_run(
    thread_id: str,
    request: RunCreate,
    user: User = Depends(get_current_user),
    tenant: Tenant = Depends(get_current_tenant),
) -> StreamingResponse:
    """Create a run, execute it, and wait for completion.

    Returns a chunked ``application/json`` response with periodic ``\\n``
    heartbeat bytes to keep the connection alive. The final chunk is the
    JSON result. Uses ``BG_JOB_TIMEOUT_SECS`` (default 1 hour) as the
    safety-net timeout.

    Sessions are managed manually (not via ``Depends``) to avoid holding a
    pool connection during the long wait.
    """
    # Session block: all pre-execution DB work (validate, create run, submit)
    async for session in get_session(tenant):
        run_id, _run, _job = await _prepare_run(session, thread_id, request, user, tenant, initial_status="pending")
        break

    # No pool connection held from here — safe for long waits
    return StreamingResponse(
        heartbeat_wait_body(
            run_id,
            thread_id,
            user.id,
            tenant_schema=tenant.schema,
            timeout=settings.worker.BG_JOB_TIMEOUT_SECS,
        ),
        media_type="application/json",
        headers={
            "Location": f"/threads/{thread_id}/runs/{run_id}/join",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )


@router.get("/threads/{thread_id}/runs/{run_id}/stream", responses={**SSE_RESPONSE, **NOT_FOUND})
async def stream_run(
    thread_id: str,
    run_id: str,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    _stream_mode: str | None = Query(None, description="Override the stream mode for this connection."),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
) -> StreamingResponse:
    """Stream an existing run's execution via SSE.

    Attach to a run that was created without streaming (e.g. via the create
    endpoint) to receive its events in real time. If the run has already
    finished, a single `end` event is emitted. Use the `Last-Event-ID`
    header to resume from a specific event after a disconnect.
    """
    logger.info(
        f"[stream_run] fetch for stream run_id={run_id} thread_id={thread_id} user={user.id} team={user.team_id}"
    )
    stmt = (
        select(RunORM)
        .join(ThreadORM, ThreadORM.thread_id == RunORM.thread_id)
        .where(
            RunORM.run_id == str(run_id),
            RunORM.thread_id == thread_id,
            RunORM.team_id == user.team_id,
        )
    )
    stmt = stmt.where(
        or_(
            RunORM.user_id == user.id,
            ThreadORM.is_shared.is_(True),
        )
    )
    run_orm = await session.scalar(stmt)
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    logger.info(
        f"[stream_run] status={run_orm.status} user={user.id} team={user.team_id} thread_id={thread_id} run_id={run_id}"
    )
    # If already terminal and no Last-Event-ID, just emit end.
    # If Last-Event-ID is present, fall through to stream_run_execution
    # which will replay missed events from the buffer before ending.
    if run_orm.status in TERMINAL_STATES and not last_event_id:
        final_status = "error" if run_orm.status == "error" else run_orm.status

        async def generate_final() -> AsyncIterator[str]:
            yield create_end_event(status=final_status)

        logger.info(f"[stream_run] starting terminal stream run_id={run_id} status={run_orm.status}")
        return StreamingResponse(
            generate_final(),
            media_type="text/event-stream",
            headers={
                **get_sse_headers(),
                "Location": f"/threads/{thread_id}/runs/{run_id}/stream",
                "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
            },
        )

    # Stream active or pending runs via broker

    # Build a lightweight Pydantic Run from ORM for streaming context (IDs already strings)
    # noinspection PyTypeChecker
    run_model = Run.model_validate(run_orm)

    return StreamingResponse(
        streaming_service.stream_run_execution(tenant, run_model, last_event_id, cancel_on_disconnect=False),
        media_type="text/event-stream",
        headers={
            **get_sse_headers(),
            "Location": f"/threads/{thread_id}/runs/{run_id}/stream",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )


@router.post(
    "/threads/{thread_id}/runs/{run_id}/cancel",
    response_model=Run,
    responses={**NOT_FOUND},
)
async def cancel_run_endpoint(
    thread_id: str,
    run_id: str,
    wait: int = Query(0, ge=0, le=1, description="Set to 1 to wait for the run task to settle before returning."),
    action: str = Query(
        "cancel",
        pattern="^(cancel|interrupt)$",
        description="Cancellation strategy: 'cancel' for hard cancel, 'interrupt' for cooperative interrupt.",
    ),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Cancel or interrupt a running execution.

    Use `action=cancel` to hard-cancel the run immediately, or
    `action=interrupt` to cooperatively interrupt (the graph can handle the
    interrupt and save partial state). Set `wait=1` to block until the
    background task has fully settled before returning the updated run.
    """
    logger.info(f"[cancel_run] fetch run run_id={run_id} thread_id={thread_id} user={user.id} team={user.team_id}")
    stmt = (
        select(RunORM)
        .join(ThreadORM, ThreadORM.thread_id == RunORM.thread_id)
        .where(
            RunORM.run_id == run_id,
            RunORM.thread_id == thread_id,
            RunORM.team_id == user.team_id,
        )
    )
    stmt = stmt.where(
        or_(
            RunORM.user_id == user.id,
            ThreadORM.is_shared.is_(True),
        )
    )
    run_orm = await session.scalar(stmt)
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    if action == "interrupt":
        logger.info(f"[cancel_run] interrupt run_id={run_id} user={user.id} team={user.team_id} thread_id={thread_id}")
        await streaming_service.interrupt_run(run_id)
        # Persist status as interrupted
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == str(run_id))
            .values(status="interrupted", updated_at=datetime.now(UTC))
        )
        await session.commit()
    else:
        logger.info(f"[cancel_run] cancel run_id={run_id} user={user.id} team={user.team_id} thread_id={thread_id}")
        await streaming_service.cancel_run(run_id)
        # Persist status as interrupted
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == str(run_id))
            .values(status="interrupted", updated_at=datetime.now(UTC))
        )
        await session.commit()

    # Optionally wait for the run to settle
    if wait:
        # Poll DB until the run reaches a terminal state (or 10s timeout).
        # This is simpler and more reliable than pub/sub for cancel-with-wait
        # since the cancel has already been issued and the status update committed.
        for _ in range(20):
            await asyncio.sleep(0.5)
            session.expire_all()  # sync method, clears cache
            fresh = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
            if fresh and fresh.status in TERMINAL_STATES:
                break

    # Reload and return updated Run (do NOT delete here; deletion is a separate endpoint)
    run_orm = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == run_id,
            RunORM.thread_id == thread_id,
            RunORM.team_id == user.team_id,
        )
    )
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found after cancellation")

    # noinspection PyTypeChecker
    return Run.model_validate(run_orm)


@router.delete(
    "/threads/{thread_id}/runs/{run_id}",
    status_code=204,
    responses={**NOT_FOUND, **CONFLICT},
)
async def delete_run(
    thread_id: str,
    run_id: str,
    force: int = Query(0, ge=0, le=1, description="Set to 1 to cancel an active run before deleting it."),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    """Delete a run record.

    If the run is active (pending or running) and `force=0`, returns 409
    Conflict. Set `force=1` to cancel the run first (best-effort) and then
    delete it. Returns 204 No Content on success.
    """
    # Authorization check (delete action on runs resource)
    ctx = build_auth_context(user, "runs", "delete")
    value = {"run_id": run_id, "thread_id": thread_id}
    await handle_event(ctx, value)
    logger.info(f"[delete_run] fetch run run_id={run_id} thread_id={thread_id} user={user.id} team={user.team_id}")
    stmt = (
        select(RunORM)
        .join(ThreadORM, ThreadORM.thread_id == RunORM.thread_id)
        .where(
            RunORM.run_id == str(run_id),
            RunORM.thread_id == thread_id,
            RunORM.team_id == user.team_id,
        )
    )
    stmt = stmt.where(
        or_(
            RunORM.user_id == user.id,
            ThreadORM.is_shared.is_(True),
        )
    )
    run_orm = await session.scalar(stmt)
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    # If active and not forcing, reject deletion
    if run_orm.status in ["pending", "running"] and not force:
        raise HTTPException(
            status_code=409,
            detail="Run is active. Retry with force=1 to cancel and delete.",
        )

    # If forcing and active, cancel first
    if force and run_orm.status in ["pending", "running"]:
        logger.info(f"[delete_run] force-cancelling active run run_id={run_id}")
        await streaming_service.cancel_run(run_id)
        # Best-effort: wait for bg task to settle
        task = active_runs.get(run_id)
        if task:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    # Delete the record
    await session.execute(
        delete(RunORM).where(
            RunORM.run_id == str(run_id),
            RunORM.thread_id == thread_id,
            RunORM.team_id == user.team_id,
        )
    )
    await session.commit()

    # Clean up active task if exists
    task = active_runs.pop(run_id, None)
    if task and not task.done():
        task.cancel()

    # 204 No Content
    return


# ---------- Control Plane: Run History (paginated, team-scoped) ----------


SORT_COLUMNS = {
    "created_at": RunORM.created_at,
    "status": RunORM.status,
    "duration_ms": RunORM.duration_ms,
    "updated_at": RunORM.updated_at,
}


@router.get("/runs", response_model=RunHistoryPage)
async def list_runs_history(
    status: str | None = Query(None),
    since: str | None = Query(None, description="ISO datetime"),
    assistant_id: str | None = Query(None, description="Assistant ID or name"),
    search: str | None = Query(None, description="Search by run_id prefix"),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Paginated run history with optional status/date/assistant/search filters."""
    base = (
        select(RunORM, AssistantORM.name)
        .outerjoin(AssistantORM, RunORM.assistant_id == AssistantORM.assistant_id)
        .where(RunORM.team_id == user.team_id)
    )
    count_base = (
        select(func.count())
        .select_from(RunORM)
        .outerjoin(AssistantORM, RunORM.assistant_id == AssistantORM.assistant_id)
        .where(RunORM.team_id == user.team_id)
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

    runs = []
    for run, name in rows:
        config = run.config_snapshot or run.config or {}
        model_name = config.get("model_name")
        mode = config.get("mode")
        runs.append(
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
                model_name=model_name,
                mode=mode,
                tool_calls_count=None,
                tools_used=None,
            )
        )
    return RunHistoryPage(runs=runs, total=total, limit=limit, offset=offset)


# ---------- Control Plane: Run Status History ----------


@router.get("/runs/{run_id}/history", response_model=list[RunStatusTransition])
async def get_run_status_history(
    run_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Status transition timeline for a specific run."""
    run = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == run_id,
            RunORM.team_id == user.team_id,
        )
    )
    if not run:
        raise HTTPException(404, f"Run '{run_id}' not found")

    result = await session.scalars(
        select(RunStatusHistory).where(RunStatusHistory.run_id == run_id).order_by(RunStatusHistory.created_at.asc())
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


# ---------- Control Plane: Run Detail ----------


@router.get("/runs/{run_id}/detail", response_model=RunDetailResponse)
async def get_run_detail(
    run_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Full detail view of a single run including input/output/config."""
    stmt = (
        select(RunORM, AssistantORM.name)
        .outerjoin(AssistantORM, RunORM.assistant_id == AssistantORM.assistant_id)
        .where(RunORM.run_id == run_id, RunORM.team_id == user.team_id)
    )
    row = (await session.execute(stmt)).first()
    if not row:
        raise HTTPException(404, f"Run '{run_id}' not found")

    run, assistant_name = row
    config = run.config_snapshot or run.config or {}
    model_name = config.get("model_name")
    mode = config.get("mode")

    return RunDetailResponse(
        run_id=run.run_id,
        thread_id=run.thread_id,
        assistant_id=run.assistant_id,
        assistant_name=assistant_name,
        status=run.status,
        error_message=run.error_message,
        duration_ms=run.duration_ms,
        current_step=run.current_step,
        created_at=run.created_at,
        updated_at=run.updated_at,
        input=run.input,
        output=run.output,
        config_snapshot=run.config_snapshot,
        model_name=model_name,
        mode=mode,
        tool_calls_count=None,
        tools_used=None,
        user_id=run.user_id,
    )


# ---------- Control Plane: Run Events Replay ----------


@router.get("/runs/{run_id}/events")
async def get_run_events(
    run_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Get stored SSE events for a run (available for ~1 hour after execution)."""
    run = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == run_id,
            RunORM.team_id == user.team_id,
        )
    )
    if not run:
        raise HTTPException(404, f"Run '{run_id}' not found")

    run_created_at = run.created_at

    result = await session.scalars(
        select(RunEventORM).where(RunEventORM.run_id == run_id).order_by(RunEventORM.seq.asc())
    )
    events = result.all()

    if not events:
        return []

    return [
        {
            "seq": evt.seq,
            "event": evt.event,
            "data": evt.data,
            "elapsed_seconds": round((evt.created_at - run_created_at).total_seconds(), 3),
            "created_at": evt.created_at.isoformat(),
        }
        for evt in events
    ]


# ---------- Control Plane: Cancel Run ----------


@router.post("/runs/{run_id}/cancel")
async def cancel_run_by_id(
    run_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Cancel a running task (team-scoped)."""
    logger.info("cancel_run_requested", run_id=run_id, user_id=user.id)
    run = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == run_id,
            RunORM.team_id == user.team_id,
        )
    )
    if not run:
        raise HTTPException(404, f"Run '{run_id}' not found")
    if run.status not in ("pending", "running"):
        raise HTTPException(409, f"Run is not active (status={run.status})")

    # Record status transition
    history = RunStatusHistory(
        run_id=run_id,
        from_status=run.status,
        to_status="interrupted",
        created_at=datetime.now(UTC),
    )
    session.add(history)

    # Update run status in DB
    await session.execute(
        update(RunORM)
        .where(RunORM.run_id == run_id)
        .values(status="interrupted", error_message="Cancelled via control plane")
    )
    await session.commit()

    # Cancel streaming and asyncio task
    await streaming_service.cancel_run(run_id)

    task = active_runs.get(run_id)
    if task and not task.done():
        task.cancel()

    logger.info("cancel_run_completed", run_id=run_id)
    return {"status": "interrupted", "run_id": run_id}
