"""Run endpoints for Agent Protocol"""

import asyncio
import contextlib
import traceback as tb_module
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from langgraph.types import Command, Send
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.auth_ctx import with_auth_ctx
from ..core.auth_deps import get_current_user
from ..core.orm import Assistant as AssistantORM
from ..core.orm import Run as RunORM
from ..core.orm import RunEvent as RunEventORM
from ..core.orm import RunStatusHistory
from ..core.orm import Thread as ThreadORM
from ..core.orm import _get_session_maker, get_session
from ..core.serializers import GeneralSerializer
from ..core.sse import create_end_event, get_sse_headers
from ..models import Run, RunCreate, RunStatus, User
from ..models.control_plane import (
    RunDetailResponse,
    RunHistoryEntry,
    RunHistoryPage,
    RunStatusTransition,
)
from ..services.langgraph_service import create_run_config, get_langgraph_service, inject_mcp_tools
from ..services.streaming_service import streaming_service
from ..utils.assistants import resolve_assistant_id
from ..utils.config_merge import merge_runtime_config
from ..utils.run_utils import _should_skip_event


router = APIRouter()

logger = structlog.getLogger(__name__)
serializer = GeneralSerializer()

# NOTE: We keep only an in-memory task registry for asyncio.Task handles.
# All run metadata/state is persisted via ORM.
active_runs: dict[str, asyncio.Task] = {}

# Default stream modes for background run execution
DEFAULT_STREAM_MODES = ["values"]


def map_command_to_langgraph(cmd: dict[str, Any]) -> Command:
    """Convert API command to LangGraph Command"""
    goto = cmd.get("goto")
    if goto is not None and not isinstance(goto, list):
        goto = [goto]

    update = cmd.get("update")
    if isinstance(update, (tuple, list)) and all(
        isinstance(t, (tuple, list)) and len(t) == 2 and isinstance(t[0], str)
        for t in update
    ):
        update = [tuple(t) for t in update]

    return Command(
        update=update,
        goto=(
            [
                it if isinstance(it, str) else Send(it["node"], it["input"])
                for it in goto
            ]
            if goto
            else None
        ),
        resume=cmd.get("resume"),
    )


async def set_thread_status(session: AsyncSession, thread_id: str, status: str) -> None:
    """Update the status column of a thread."""
    await session.execute(
        update(ThreadORM)
        .where(ThreadORM.thread_id == thread_id)
        .values(status=status, updated_at=datetime.now(UTC))
    )
    await session.commit()


async def update_thread_metadata(
    session: AsyncSession,
    thread_id: str,
    assistant_id: str,
    graph_id: str,
    is_shared: bool,
    input: dict[str, Any],
) -> None:
    """Update thread metadata with assistant and graph information (dialect agnostic)."""
    # Read-modify-write to avoid DB-specific JSON concat operators
    thread = await session.scalar(
        select(ThreadORM).where(ThreadORM.thread_id == thread_id)
    )
    if not thread:
        raise HTTPException(404, f"Thread '{thread_id}' not found for metadata update")
    md = dict(getattr(thread, "metadata_json", {}) or {})

    if not md.get("thread_name"):
        messages = input.get("messages") or []
        first = messages[0] if messages else {}

        content = first.get("content") or []

        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                break

        if text:
            md["thread_name"] = text[:60]

    md.update(
        {
            "assistant_id": str(assistant_id),
            "graph_id": graph_id,
        }
    )
    await session.execute(
        update(ThreadORM)
        .where(ThreadORM.thread_id == thread_id)
        .values(
            metadata_json=md,
            updated_at=datetime.now(UTC),
            assistant_id=assistant_id,
            is_shared=is_shared,
        )
    )
    await session.commit()


async def _validate_resume_command(
    session: AsyncSession, thread_id: str, command: dict[str, Any] | None
) -> None:
    """Validate resume command requirements."""
    if command and command.get("resume") is not None:
        # Check if thread exists and is in interrupted state
        thread_stmt = select(ThreadORM).where(ThreadORM.thread_id == thread_id)
        thread = await session.scalar(thread_stmt)
        if not thread:
            raise HTTPException(404, f"Thread '{thread_id}' not found")
        if thread.status != "interrupted":
            raise HTTPException(
                400, "Cannot resume: thread is not in interrupted state"
            )


@router.post("/threads/{thread_id}/runs", response_model=Run)
async def create_run(
    thread_id: str,
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Create and execute a new run (persisted)."""

    # Validate resume command requirements early
    await _validate_resume_command(session, thread_id, request.command)

    run_id = str(uuid4())

    # Get LangGraph service
    langgraph_service = get_langgraph_service()
    logger.info(
        f"[create_run] scheduling background task run_id={run_id} thread_id={thread_id} user={user.id} team={user.team_id}"
    )

    # Validate assistant exists and get its graph_id. If a graph_id was provided
    # instead of an assistant UUID, map it deterministically and fall back to the
    # default assistant created at startup.
    requested_id = str(request.assistant_id)
    available_graphs = langgraph_service.list_graphs()
    resolved_assistant_id = resolve_assistant_id(requested_id, available_graphs)

    assistant_stmt = select(AssistantORM).where(
        AssistantORM.assistant_id == resolved_assistant_id,
    )
    assistant = await session.scalar(assistant_stmt)
    if not assistant:
        raise HTTPException(404, f"Assistant '{request.assistant_id}' not found")

    if assistant.deleted_at is not None:
        raise HTTPException(400, f"Assistant '{request.assistant_id}' has been deleted")

    if not assistant.enabled:
        raise HTTPException(
            403,
            f"Assistant '{request.assistant_id}' is currently disabled. Please enable it to create runs.",
        )

    config = merge_runtime_config(assistant.config, request.config)
    context = assistant.context

    # Validate the assistant's graph exists
    available_graphs = langgraph_service.list_graphs()
    if assistant.graph_id not in available_graphs:
        raise HTTPException(
            404, f"Graph '{assistant.graph_id}' not found for assistant"
        )

    is_shared = config.get("configurable", {}).get("share_new_chats_by_default", False)

    # Mark thread as busy and update metadata with assistant/graph info
    await set_thread_status(session, thread_id, "busy")

    await update_thread_metadata(
        session,
        thread_id,
        assistant.assistant_id,
        assistant.graph_id,
        is_shared,
        request.input,
    )

    # Persist run record via ORM model in core.orm (Run table)
    now = datetime.now(UTC)
    run_orm = RunORM(
        run_id=run_id,  # explicitly set (DB can also default-generate if omitted)
        thread_id=thread_id,
        assistant_id=resolved_assistant_id,
        status="pending",
        input=request.input or {},
        config=config,
        context=context,
        user_id=user.id,
        team_id=user.team_id,
        created_at=now,
        updated_at=now,
        output=None,
        error_message=None,
    )
    session.add(run_orm)
    await session.commit()

    # Build response from ORM -> Pydantic
    run = Run(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id=resolved_assistant_id,
        status="pending",
        input=request.input or {},
        user_id=user.id,
        team_id=user.team_id,
        created_at=now,
        updated_at=now,
        output=None,
        error_message=None,
    )

    # Start execution asynchronously
    # Don't pass the session to avoid transaction conflicts
    task = asyncio.create_task(
        execute_run_async(
            run_id,
            thread_id,
            assistant.graph_id,
            request.input or {},
            user,
            config,
            context,
            request.stream_mode,
            None,  # Don't pass session to avoid conflicts
            request.checkpoint,
            request.command,
            request.interrupt_before,
            request.interrupt_after,
            request.multitask_strategy,
            request.stream_subgraphs,
        )
    )
    logger.info(
        f"[create_run] background task created task_id={id(task)} for run_id={run_id}"
    )
    active_runs[run_id] = task

    return run


@router.post("/threads/{thread_id}/runs/stream")
async def create_and_stream_run(
    thread_id: str,
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Create a new run and stream its execution - persisted + SSE."""

    # Validate resume command requirements early
    await _validate_resume_command(session, thread_id, request.command)

    run_id = str(uuid4())

    # Get LangGraph service
    langgraph_service = get_langgraph_service()
    logger.info(
        f"[create_and_stream_run] scheduling background task run_id={run_id} thread_id={thread_id} user={user.id} team={user.team_id}"
    )

    # Validate assistant exists and get its graph_id. Allow passing a graph_id
    # by mapping it to a deterministic assistant ID.
    requested_id = str(request.assistant_id)
    available_graphs = langgraph_service.list_graphs()

    resolved_assistant_id = resolve_assistant_id(requested_id, available_graphs)

    assistant_stmt = select(AssistantORM).where(
        AssistantORM.assistant_id == resolved_assistant_id,
    )
    assistant = await session.scalar(assistant_stmt)
    if not assistant:
        raise HTTPException(404, f"Assistant '{request.assistant_id}' not found")

    if assistant.deleted_at is not None:
        raise HTTPException(400, f"Assistant '{request.assistant_id}' has been deleted")

    if not assistant.enabled:
        raise HTTPException(
            403,
            f"Assistant '{request.assistant_id}' is currently disabled. Please enable it to create runs.",
        )

    config = merge_runtime_config(assistant.config, request.config)
    context = assistant.context

    # Validate the assistant's graph exists
    available_graphs = langgraph_service.list_graphs()
    if assistant.graph_id not in available_graphs:
        raise HTTPException(
            404, f"Graph '{assistant.graph_id}' not found for assistant"
        )

    is_shared = config.get("configurable", {}).get("share_new_chats_by_default", False)

    # Mark thread as busy and update metadata with assistant/graph info
    await set_thread_status(session, thread_id, "busy")
    await update_thread_metadata(
        session,
        thread_id,
        assistant.assistant_id,
        assistant.graph_id,
        is_shared,
        request.input,
    )

    # Persist run record
    now = datetime.now(UTC)
    run_orm = RunORM(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id=resolved_assistant_id,
        status="streaming",
        input=request.input or {},
        config=config,
        context=context,
        user_id=user.id,
        team_id=user.team_id,
        created_at=now,
        updated_at=now,
        output=None,
        error_message=None,
    )
    session.add(run_orm)
    await session.commit()

    # Build response model for stream context
    run = Run(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id=resolved_assistant_id,
        status="streaming",
        input=request.input or {},
        user_id=user.id,
        team_id=user.team_id,
        created_at=now,
        updated_at=now,
        output=None,
        error_message=None,
    )

    # Start background execution that will populate the broker
    # Don't pass the session to avoid transaction conflicts
    task = asyncio.create_task(
        execute_run_async(
            run_id,
            thread_id,
            assistant.graph_id,
            request.input or {},
            user,
            config,
            context,
            request.stream_mode,
            None,  # Don't pass session to avoid conflicts
            request.checkpoint,
            request.command,
            request.interrupt_before,
            request.interrupt_after,
            request.multitask_strategy,
            request.stream_subgraphs,
        )
    )
    logger.info(
        f"[create_and_stream_run] background task created task_id={id(task)} for run_id={run_id}"
    )
    active_runs[run_id] = task

    # Extract requested stream mode(s)
    stream_mode = request.stream_mode
    if not stream_mode and config and "stream_mode" in config:
        stream_mode = config["stream_mode"]

    # Stream immediately from broker (which will also include replay of any early events)
    cancel_on_disconnect = (request.on_disconnect or "continue").lower() == "cancel"

    return StreamingResponse(
        streaming_service.stream_run_execution(
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


@router.get("/threads/{thread_id}/runs/{run_id}", response_model=Run)
async def get_run(
    thread_id: str,
    run_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Get run by ID (persisted)."""
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
    logger.info(
        f"[get_run] querying DB run_id={run_id} thread_id={thread_id} user={user.id} team={user.team_id}"
    )
    run_orm = await session.scalar(stmt)
    if not run_orm:
        raise HTTPException(404, f"Run '{run_id}' not found")

    # Refresh to ensure we have the latest data (in case background task updated it)
    await session.refresh(run_orm)

    logger.info(
        f"[get_run] found run status={run_orm.status} user={user.id} thread_id={thread_id} run_id={run_id}"
    )
    # Convert to Pydantic
    # noinspection PyTypeChecker
    return Run.model_validate(
        {c.name: getattr(run_orm, c.name) for c in run_orm.__table__.columns}
    )


@router.get("/threads/{thread_id}/runs", response_model=list[Run])
async def list_runs(
    thread_id: str,
    limit: int = Query(10, ge=1, description="Maximum number of runs to return"),
    offset: int = Query(0, ge=0, description="Number of runs to skip for pagination"),
    status: str | None = Query(None, description="Filter by run status"),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> list[Run]:
    """List runs for a specific thread (persisted)."""

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

    runs = [
        Run.model_validate({c.name: getattr(r, c.name) for c in r.__table__.columns})
        for r in rows
    ]
    logger.info(f"[list_runs] total={len(runs)} user={user.id} thread_id={thread_id}")
    return runs


@router.patch("/threads/{thread_id}/runs/{run_id}")
async def update_run(
    thread_id: str,
    run_id: str,
    request: RunStatus,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """Update run status (for cancellation/interruption, persisted)."""
    logger.info(
        f"[update_run] fetch for update run_id={run_id} thread_id={thread_id} user={user.id}"
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

    # Handle interruption/cancellation

    if request.status == "cancelled":
        logger.info(
            f"[update_run] cancelling run_id={run_id} user={user.id} team={user.team_id} thread_id={thread_id}"
        )
        await streaming_service.cancel_run(run_id)
        logger.info(f"[update_run] set DB status=cancelled run_id={run_id}")
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == str(run_id))
            .values(status="cancelled", updated_at=datetime.now(UTC))
        )
        await session.commit()
        logger.info(f"[update_run] commit done (cancelled) run_id={run_id}")
    elif request.status == "interrupted":
        logger.info(
            f"[update_run] interrupt run_id={run_id} user={user.id} team={user.team_id} thread_id={thread_id}"
        )
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
    if run_orm:
        # Refresh to ensure we have the latest data after our own update
        await session.refresh(run_orm)
    # noinspection PyTypeChecker
    return Run.model_validate(
        {c.name: getattr(run_orm, c.name) for c in run_orm.__table__.columns}
    )


@router.get("/threads/{thread_id}/runs/{run_id}/join")
async def join_run(
    thread_id: str,
    run_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Join a run (wait for completion and return final output) - persisted."""
    # Get run and validate it exists
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

    # If already completed, return output immediately
    if run_orm.status in ["completed", "failed", "cancelled"]:
        # Refresh to ensure we have the latest data
        await session.refresh(run_orm)
        output = getattr(run_orm, "output", None) or {}
        return output

    # Wait for a background task to complete
    task = active_runs.get(run_id)
    if task:
        try:
            await asyncio.wait_for(task, timeout=30.0)
        except TimeoutError:
            # Task is taking too long, but that's okay - we'll check DB status
            pass
        except asyncio.CancelledError:
            # Task was cancelled, that's also okay
            pass

    # Return final output from a database
    run_orm = await session.scalar(select(RunORM).where(RunORM.run_id == run_id))
    if run_orm:
        await session.refresh(run_orm)  # Refresh to get the latest data from DB
    output = getattr(run_orm, "output", None) or {}
    return output


@router.post("/threads/{thread_id}/runs/wait")
async def wait_for_run(
    thread_id: str,
    request: RunCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Create a run, execute it, and wait for completion (Agent Protocol).

    This endpoint combines run creation and execution with synchronous waiting.
    Returns the final output directly (not the Run object).

    Compatible with LangGraph SDK's runs.wait() method and Agent Protocol spec.
    """
    # Validate resume command requirements early
    await _validate_resume_command(session, thread_id, request.command)

    run_id = str(uuid4())

    # Get LangGraph service
    langgraph_service = get_langgraph_service()
    logger.info(
        f"[wait_for_run] creating run run_id={run_id} thread_id={thread_id} user={user.id} team={user.team_id}"
    )

    # Validate assistant exists and get its graph_id
    requested_id = str(request.assistant_id)
    available_graphs = langgraph_service.list_graphs()
    resolved_assistant_id = resolve_assistant_id(requested_id, available_graphs)

    assistant_stmt = select(AssistantORM).where(
        AssistantORM.assistant_id == resolved_assistant_id,
        AssistantORM.team_id == user.team_id,
    )
    assistant = await session.scalar(assistant_stmt)
    if not assistant:
        raise HTTPException(404, f"Assistant '{request.assistant_id}' not found")

    config = merge_runtime_config(assistant.config, request.config)
    context = assistant.context

    # Validate the assistant's graph exists
    available_graphs = langgraph_service.list_graphs()
    if assistant.graph_id not in available_graphs:
        raise HTTPException(
            404, f"Graph '{assistant.graph_id}' not found for assistant"
        )

    is_shared = config.get("configurable", {}).get("share_new_chats_by_default", False)

    # Mark thread as busy and update metadata with assistant/graph info
    await set_thread_status(session, thread_id, "busy")
    await update_thread_metadata(
        session,
        thread_id,
        assistant.assistant_id,
        assistant.graph_id,
        is_shared,
        request.input,
    )

    # Persist run record
    now = datetime.now(UTC)
    run_orm = RunORM(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id=resolved_assistant_id,
        status="pending",
        input=request.input or {},
        config=config,
        context=context,
        user_id=user.id,
        team_id=user.team_id,
        created_at=now,
        updated_at=now,
        output=None,
        error_message=None,
    )
    session.add(run_orm)
    await session.commit()

    # Start execution asynchronously
    task = asyncio.create_task(
        execute_run_async(
            run_id,
            thread_id,
            assistant.graph_id,
            request.input or {},
            user,
            config,
            context,
            request.stream_mode,
            None,  # Don't pass session to avoid conflicts
            request.checkpoint,
            request.command,
            request.interrupt_before,
            request.interrupt_after,
            request.multitask_strategy,
            request.stream_subgraphs,
        )
    )
    logger.info(
        f"[wait_for_run] background task created task_id={id(task)} for run_id={run_id}"
    )
    active_runs[run_id] = task

    # Wait for task to complete with timeout
    try:
        await asyncio.wait_for(task, timeout=300.0)  # 5 minute timeout
    except TimeoutError:
        logger.warning(f"[wait_for_run] timeout waiting for run_id={run_id}")
        # Don't raise, just return the current state
    except asyncio.CancelledError:
        logger.info(f"[wait_for_run] cancelled run_id={run_id}")
        # Task was cancelled, continue to return the final state
    except Exception as e:
        logger.error(f"[wait_for_run] exception in run_id={run_id}: {e}")
        # Exception already handled by execute_run_async

    # Get final output from a database
    run_orm = await session.scalar(
        select(RunORM).where(
            RunORM.run_id == run_id,
            RunORM.thread_id == thread_id,
            RunORM.user_id == user.id,
        )
    )
    if not run_orm:
        raise HTTPException(500, f"Run '{run_id}' disappeared during execution")

    await session.refresh(run_orm)

    # Return output based on final status
    if run_orm.status == "completed":
        return run_orm.output or {}
    elif run_orm.status == "failed":
        # For failed runs, still return output if available, but log the error
        logger.error(
            f"[wait_for_run] run failed run_id={run_id} error={run_orm.error_message}"
        )
        return run_orm.output or {}
    elif run_orm.status == "interrupted":
        # Return partial output for interrupted runs
        return run_orm.output or {}
    elif run_orm.status == "cancelled":
        return run_orm.output or {}
    else:
        # Still pending/running after timeout
        return run_orm.output or {}


# TODO: check if this method is actually required because the implementation doesn't seem correct.
@router.get("/threads/{thread_id}/runs/{run_id}/stream")
async def stream_run(
    thread_id: str,
    run_id: str,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    _stream_mode: str | None = Query(None),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Stream run execution with SSE and reconnection support - persisted metadata."""
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
    # If already terminal, emit a final end event
    if run_orm.status in ["completed", "failed", "cancelled"]:

        async def generate_final() -> AsyncIterator[str]:
            yield create_end_event()

        logger.info(
            f"[stream_run] starting terminal stream run_id={run_id} status={run_orm.status}"
        )
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
    run_model = Run.model_validate(
        {c.name: getattr(run_orm, c.name) for c in run_orm.__table__.columns}
    )

    return StreamingResponse(
        streaming_service.stream_run_execution(
            run_model, last_event_id, cancel_on_disconnect=False
        ),
        media_type="text/event-stream",
        headers={
            **get_sse_headers(),
            "Location": f"/threads/{thread_id}/runs/{run_id}/stream",
            "Content-Location": f"/threads/{thread_id}/runs/{run_id}",
        },
    )


@router.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run_endpoint(
    thread_id: str,
    run_id: str,
    wait: int = Query(
        0, ge=0, le=1, description="Whether to wait for the run task to settle"
    ),
    action: str = Query(
        "cancel", pattern="^(cancel|interrupt)$", description="Cancellation action"
    ),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> Run:
    """
    Cancel or interrupt a run (client-compatible endpoint).

    Matches client usage:
      POST /v1/threads/{thread_id}/runs/{run_id}/cancel?wait=0&action=interrupt

    - action=cancel => hard cancel
    - action=interrupt => cooperative interrupt if supported
    - wait=1 => await a background task to finish settling
    """
    logger.info(
        f"[cancel_run] fetch run run_id={run_id} thread_id={thread_id} user={user.id} team={user.team_id}"
    )
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
        logger.info(
            f"[cancel_run] interrupt run_id={run_id} user={user.id} team={user.team_id} thread_id={thread_id}"
        )
        await streaming_service.interrupt_run(run_id)
        # Persist status as interrupted
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == str(run_id))
            .values(status="interrupted", updated_at=datetime.now(UTC))
        )
        await session.commit()
    else:
        logger.info(
            f"[cancel_run] cancel run_id={run_id} user={user.id} team={user.team_id} thread_id={thread_id}"
        )
        await streaming_service.cancel_run(run_id)
        # Persist status as cancelled
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == str(run_id))
            .values(status="cancelled", updated_at=datetime.now(UTC))
        )
        await session.commit()

    # Optionally wait for a background task
    if wait:
        task = active_runs.get(run_id)
        if task:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

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
    return Run.model_validate(
        {c.name: getattr(run_orm, c.name) for c in run_orm.__table__.columns}
    )


async def execute_run_async(
    run_id: str,
    thread_id: str,
    graph_id: str,
    input_data: dict,
    user: User,
    config: dict | None = None,
    context: dict | None = None,
    stream_mode: list[str] | None = None,
    session: AsyncSession | None = None,
    checkpoint: dict | None = None,
    command: dict[str, Any] | None = None,
    interrupt_before: str | list[str] | None = None,
    interrupt_after: str | list[str] | None = None,
    _multitask_strategy: str | None = None,
    subgraphs: bool | None = False,
) -> None:
    """Execute run asynchronously in background using streaming to capture all events"""  # Use provided session or get a new one
    if session is None:
        maker = _get_session_maker()
        session = maker()

    # Normalize stream_mode once here for all callers/endpoints.
    # Accept "messages-tuple" as an alias of "messages".
    def _normalize_mode(mode: str | None) -> str | None:
        return (
            "messages" if isinstance(mode, str) and mode == "messages-tuple" else mode
        )

    if isinstance(stream_mode, list):
        stream_mode = [_normalize_mode(m) for m in stream_mode]
    else:
        stream_mode = _normalize_mode(stream_mode)

    # --- Run metrics tracking ---
    tool_calls_count = 0
    tools_used_set: set[str] = set()
    config_snapshot: dict | None = None

    try:
        # Update status
        await update_run_status(run_id, "running", session=session)

        # Get graph and execute
        langgraph_service = get_langgraph_service()

        graph = await langgraph_service.get_graph(graph_id)

        run_config = create_run_config(
            run_id, thread_id, user, config or {}, checkpoint
        )

        # Handle human-in-the-loop fields
        if interrupt_before is not None:
            run_config["interrupt_before"] = (
                interrupt_before
                if isinstance(interrupt_before, list)
                else [interrupt_before]
            )
        if interrupt_after is not None:
            run_config["interrupt_after"] = (
                interrupt_after
                if isinstance(interrupt_after, list)
                else [interrupt_after]
            )

        # Note: multitask_strategy is handled at the run creation level, not execution level
        # It controls concurrent run behavior, not graph execution behavior

        # Capture config snapshot for metrics
        config_snapshot = run_config.get("configurable", {})
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == str(run_id))
            .values(config_snapshot=config_snapshot)
        )
        await session.commit()

        # Pre-load MCP tools with persistent sessions for this run.
        mcp_stack = contextlib.AsyncExitStack()
        try:
            run_config = await inject_mcp_tools(graph, run_config, mcp_stack)
        except Exception as e:
            logger.warning("Failed to pre-load MCP tools for run %s: %s", run_id, e)

        # Determine input for execution (either input_data or command)
        if command is not None:
            # When command is provided, it replaces input entirely (LangGraph API behavior)
            if isinstance(command, dict):
                execution_input = map_command_to_langgraph(command)
            else:
                # Direct resume value (backward compatibility)
                execution_input = Command(resume=command)
        else:
            # No command, use regular input
            execution_input = input_data

        # Execute using streaming to capture events for later replay
        event_counter = 0
        final_output = None
        has_interrupt = False

        # Prepare stream modes for execution
        if stream_mode is None:
            final_stream_modes = DEFAULT_STREAM_MODES.copy()
        elif isinstance(stream_mode, str):
            final_stream_modes = [stream_mode]
        else:
            final_stream_modes = stream_mode.copy()

        # Ensure interrupt events are captured by including updates mode
        # Track whether updates were explicitly requested by the user
        user_requested_updates = "updates" in final_stream_modes
        if not user_requested_updates:
            final_stream_modes.append("updates")

        only_interrupt_updates = False

        async with mcp_stack, with_auth_ctx(user, []):
            async for raw_event in graph.astream(
                execution_input,
                config=run_config,
                context=context,
                subgraphs=subgraphs,
                stream_mode=final_stream_modes,
            ):
                # Skip events that contain langsmith:nostream tag
                if _should_skip_event(raw_event):
                    continue

                event_counter += 1
                event_id = f"{run_id}_event_{event_counter}"

                # Forward to broker for live consumers
                await streaming_service.put_to_broker(
                    run_id,
                    event_id,
                    raw_event,
                    only_interrupt_updates=only_interrupt_updates,
                )
                # Store for replay
                await streaming_service.store_event_from_raw(
                    run_id,
                    event_id,
                    raw_event,
                    only_interrupt_updates=only_interrupt_updates,
                )

                # Check for interrupt in this event
                event_data = None
                if isinstance(raw_event, tuple) and len(raw_event) >= 2:
                    event_data = raw_event[1]
                elif not isinstance(raw_event, tuple):
                    event_data = raw_event

                if isinstance(event_data, dict) and "__interrupt__" in event_data:
                    has_interrupt = True

                # Track current_step from updates events
                if (
                    isinstance(raw_event, tuple)
                    and len(raw_event) >= 2
                    and raw_event[0] == "updates"
                    and isinstance(raw_event[1], dict)
                ):
                    node_names = [
                        k for k in raw_event[1] if not k.startswith("__")
                    ]
                    if node_names:
                        await session.execute(
                            update(RunORM)
                            .where(RunORM.run_id == str(run_id))
                            .values(current_step=node_names[0])
                        )
                        await session.commit()

                # Track tool calls from updates events
                if (
                    isinstance(raw_event, tuple)
                    and len(raw_event) >= 2
                    and raw_event[0] == "updates"
                    and isinstance(raw_event[1], dict)
                ):
                    for node_name, node_data in raw_event[1].items():
                        if node_name.startswith("__"):
                            continue
                        if isinstance(node_data, dict):
                            msgs = node_data.get("messages", [])
                            if isinstance(msgs, list):
                                for msg in msgs:
                                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                                        for tc in msg.tool_calls:
                                            if isinstance(tc, dict) and "name" in tc:
                                                tool_calls_count += 1
                                                tools_used_set.add(tc["name"])

                # Track the final output
                if isinstance(raw_event, tuple):
                    if len(raw_event) >= 2 and raw_event[0] == "values":
                        final_output = raw_event[1]
                elif not isinstance(raw_event, tuple):
                    # Non-tuple events are values mode
                    final_output = raw_event

        # Save tool metrics
        await session.execute(
            update(RunORM)
            .where(RunORM.run_id == str(run_id))
            .values(
                tool_calls_count=tool_calls_count,
                tools_used=list(tools_used_set) if tools_used_set else None,
            )
        )
        await session.commit()

        if has_interrupt:
            await update_run_status(
                run_id, "interrupted", output=final_output or {}, session=session
            )
            if not session:
                raise RuntimeError(
                    f"No database session available to update thread {thread_id} status"
                )
            await set_thread_status(session, thread_id, "interrupted")

            # Signal interrupt completion to broker with explicit end event
            event_counter += 1
            end_event_id = f"{run_id}_event_{event_counter}"
            await streaming_service.put_to_broker(
                run_id, end_event_id, ("end", {"status": "interrupted"})
            )
            await streaming_service.store_event_from_raw(
                run_id, end_event_id, ("end", {"status": "interrupted"})
            )

        else:
            # Update with results
            await update_run_status(
                run_id, "completed", output=final_output or {}, session=session
            )
            # Mark thread back to idle
            if not session:
                raise RuntimeError(
                    f"No database session available to update thread {thread_id} status"
                )
            await set_thread_status(session, thread_id, "idle")

            # Signal successful completion to broker with explicit end event
            event_counter += 1
            end_event_id = f"{run_id}_event_{event_counter}"
            await streaming_service.put_to_broker(
                run_id, end_event_id, ("end", {"status": "completed"})
            )
            await streaming_service.store_event_from_raw(
                run_id, end_event_id, ("end", {"status": "completed"})
            )
            logger.info("run_completed", run_id=run_id, thread_id=thread_id, event_count=event_counter)

    except asyncio.CancelledError:
        logger.info("run_cancelled", run_id=run_id, thread_id=thread_id)
        # Store empty output to avoid JSON serialization issues
        await update_run_status(run_id, "cancelled", output={}, session=session)
        if not session:
            raise RuntimeError(
                f"No database session available to update thread {thread_id} status"
            ) from None
        await set_thread_status(session, thread_id, "idle")
        # Signal cancellation to broker
        await streaming_service.signal_run_cancelled(run_id)
        raise
    except Exception as e:
        logger.error("run_failed", run_id=run_id, thread_id=thread_id, error=str(e), exc_info=True)
        # Store empty output to avoid JSON serialization issues
        await update_run_status(
            run_id, "failed", output={}, error=str(e), session=session
        )
        if not session:
            raise RuntimeError(
                f"No database session available to update thread {thread_id} status"
            ) from None
        await set_thread_status(session, thread_id, "idle")
        # Signal error to broker
        await streaming_service.signal_run_error(run_id, str(e))
        raise
    finally:
        # Clean up broker
        await streaming_service.cleanup_run(run_id)
        active_runs.pop(run_id, None)


async def update_run_status(
    run_id: str,
    status: str,
    output: Any = None,
    error: str | None = None,
    session: AsyncSession | None = None,
) -> None:
    """Update run status in database (persisted). If session not provided, opens a short-lived session."""
    owns_session = False
    if session is None:
        maker = _get_session_maker()
        session = maker()  # type: ignore[assignment]
        owns_session = True
    try:
        now = datetime.now(UTC)
        values: dict[str, Any] = {"status": status, "updated_at": now}
        if output is not None:
            # Serialize output to ensure JSON compatibility
            try:
                serialized_output = serializer.serialize(output)
                values["output"] = serialized_output
            except Exception as e:
                logger.warning("output_serialization_failed", run_id=run_id, error=str(e))
                # noinspection PyTypeChecker
                values["output"] = {
                    "error": "Output serialization failed",
                    "original_type": str(type(output)),
                }
        if error is not None:
            values["error_message"] = error

        # Fetch previous status for history tracking and duration calc
        run_orm = await session.scalar(
            select(RunORM).where(RunORM.run_id == str(run_id))
        )
        from_status = run_orm.status if run_orm else None

        # Compute duration_ms on terminal statuses
        if status in ("completed", "failed", "cancelled", "interrupted") and run_orm:
            duration = now - run_orm.created_at
            values["duration_ms"] = int(duration.total_seconds() * 1000)

        await session.execute(
            update(RunORM).where(RunORM.run_id == str(run_id)).values(**values)
        )  # type: ignore[arg-type]

        # Record status transition history
        tb_str = None
        if error and status == "failed":
            tb_str = tb_module.format_exc()
            if tb_str == "NoneType: None\n":
                tb_str = None
        session.add(RunStatusHistory(
            run_id=run_id,
            from_status=from_status,
            to_status=status,
            error_message=error,
            traceback=tb_str,
        ))

        await session.commit()
        logger.info(
            "run_status_updated",
            run_id=run_id,
            from_status=from_status,
            to_status=status,
            duration_ms=values.get("duration_ms"),
        )
    finally:
        # Close only if we created it here
        if owns_session:
            await session.close()  # type: ignore[func-returns-value]


@router.delete("/threads/{thread_id}/runs/{run_id}", status_code=204)
async def delete_run(
    thread_id: str,
    run_id: str,
    force: int = Query(
        0, ge=0, le=1, description="Force cancel active run before delete (1=yes)"
    ),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    """
    Delete a run record.

    Behavior:
    - If the run is active (pending/running/streaming) and force=0, returns 409 Conflict.
    - If force=1 and the run is active, cancels it first (best-effort) and then deletes.
    - Always returns 204 No Content on successful deletion.
    """
    logger.info(
        f"[delete_run] fetch run run_id={run_id} thread_id={thread_id} user={user.id} team={user.team_id}"
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

    # If active and not forcing, reject deletion
    if run_orm.status in ["pending", "running", "streaming"] and not force:
        raise HTTPException(
            status_code=409,
            detail="Run is active. Retry with force=1 to cancel and delete.",
        )

    # If forcing and active, cancel first
    if force and run_orm.status in ["pending", "running", "streaming"]:
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
        select(RunEventORM)
        .where(RunEventORM.run_id == run_id)
        .order_by(RunEventORM.seq.asc())
    )
    events = result.all()

    if not events:
        return []

    return [
        {
            "seq": evt.seq,
            "event": evt.event,
            "data": evt.data,
            "elapsed_seconds": round(
                (evt.created_at - run_created_at).total_seconds(), 3
            ),
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
