"""Endpoints for managing Workflow Schedules (cron-based auto-execution)."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from croniter import croniter
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.auth_deps import auth_dependency, get_current_user
from ..core.orm import Workflow, WorkflowSchedule, get_session
from ..models.auth import User

router = APIRouter(tags=["Workflow Schedules"], dependencies=auth_dependency)


def _scope_workflow_query(query, user: User):
    query = query.where(Workflow.team_id == user.team_id)
    if not user.is_admin:
        query = query.where(Workflow.user_id == user.id)
    return query


def _scope_schedule_query(query, user: User):
    query = query.where(WorkflowSchedule.team_id == user.team_id)
    if not user.is_admin:
        query = query.where(WorkflowSchedule.user_id == user.id)
    return query


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ScheduleCreate(BaseModel):
    workflow_id: str
    cron_expression: str
    timezone: str = "UTC"
    input_data: dict | None = None
    is_enabled: bool = True


class ScheduleUpdate(BaseModel):
    cron_expression: str | None = None
    timezone: str | None = None
    input_data: dict | None = None
    is_enabled: bool | None = None


class ScheduleResponse(BaseModel):
    id: str
    workflow_id: str
    team_id: str
    user_id: str
    cron_expression: str
    timezone: str
    is_enabled: bool
    input_data: dict | None
    last_run_at: str | None
    next_run_at: str | None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_next_run(cron_expression: str, timezone: str) -> datetime:
    """Compute the next run time in UTC from a cron expression + timezone."""
    tz = ZoneInfo(timezone)
    now_local = datetime.now(UTC).astimezone(tz)
    cron = croniter(cron_expression, now_local)
    return cron.get_next(datetime).astimezone(UTC)


def _validate_cron(cron_expression: str) -> None:
    if not croniter.is_valid(cron_expression):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid cron expression: {cron_expression}",
        )


def _validate_timezone(timezone: str) -> None:
    try:
        ZoneInfo(timezone)
    except (KeyError, Exception):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid timezone: {timezone}",
        )


def _to_response(schedule: WorkflowSchedule) -> ScheduleResponse:
    return ScheduleResponse(
        id=schedule.id,
        workflow_id=schedule.workflow_id,
        team_id=schedule.team_id,
        user_id=schedule.user_id,
        cron_expression=schedule.cron_expression,
        timezone=schedule.timezone,
        is_enabled=schedule.is_enabled,
        input_data=schedule.input_data,
        last_run_at=schedule.last_run_at.isoformat() if schedule.last_run_at else None,
        next_run_at=schedule.next_run_at.isoformat() if schedule.next_run_at else None,
        created_at=schedule.created_at.isoformat(),
        updated_at=schedule.updated_at.isoformat(),
    )


async def _get_schedule_or_404(session: AsyncSession, schedule_id: str, user: User) -> WorkflowSchedule:
    result = await session.execute(
        _scope_schedule_query(
            select(WorkflowSchedule).where(
                WorkflowSchedule.id == schedule_id,
            ),
            user,
        )
    )
    schedule = result.scalar_one_or_none()
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return schedule


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/workflow-schedules", response_model=ScheduleResponse, status_code=201)
async def create_schedule(
    body: ScheduleCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Create a cron schedule for a workflow."""
    _validate_cron(body.cron_expression)
    _validate_timezone(body.timezone)

    # Verify workflow exists and belongs to team
    result = await session.execute(
        _scope_workflow_query(
            select(Workflow).where(
                Workflow.id == body.workflow_id,
                Workflow.deleted_at.is_(None),
            ),
            user,
        )
    )
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if not workflow.is_active:
        raise HTTPException(status_code=400, detail="Workflow is not active")

    # Check for existing active schedule (DB enforces via unique partial index too)
    existing = await session.execute(
        select(WorkflowSchedule).where(
            WorkflowSchedule.workflow_id == body.workflow_id,
            WorkflowSchedule.is_enabled.is_(True),
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail="Workflow already has an active schedule",
        )

    schedule = WorkflowSchedule(
        workflow_id=body.workflow_id,
        team_id=user.team_id,
        user_id=user.id,
        cron_expression=body.cron_expression,
        timezone=body.timezone,
        is_enabled=body.is_enabled,
        input_data=body.input_data,
        next_run_at=_compute_next_run(body.cron_expression, body.timezone) if body.is_enabled else None,
    )
    session.add(schedule)
    await session.commit()
    await session.refresh(schedule)
    return _to_response(schedule)


@router.get("/workflow-schedules", response_model=list[ScheduleResponse])
async def list_schedules(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    workflow_id: str | None = None,
):
    """List workflow schedules for the current team."""
    query = _scope_schedule_query(select(WorkflowSchedule), user)
    if workflow_id:
        query = query.where(WorkflowSchedule.workflow_id == workflow_id)

    query = query.order_by(WorkflowSchedule.created_at.desc())
    result = await session.execute(query)
    return [_to_response(s) for s in result.scalars().all()]


@router.get("/workflow-schedules/{schedule_id}", response_model=ScheduleResponse)
async def get_schedule(
    schedule_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Get a single schedule by ID."""
    schedule = await _get_schedule_or_404(session, schedule_id, user)
    return _to_response(schedule)


@router.put("/workflow-schedules/{schedule_id}", response_model=ScheduleResponse)
async def update_schedule(
    schedule_id: str,
    body: ScheduleUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Update a workflow schedule."""
    schedule = await _get_schedule_or_404(session, schedule_id, user)

    if body.cron_expression is not None:
        _validate_cron(body.cron_expression)
        schedule.cron_expression = body.cron_expression

    if body.timezone is not None:
        _validate_timezone(body.timezone)
        schedule.timezone = body.timezone

    if body.input_data is not None:
        schedule.input_data = body.input_data

    if body.is_enabled is not None:
        schedule.is_enabled = body.is_enabled

    schedule.updated_at = datetime.now(UTC)

    # Recompute next_run_at if schedule is enabled
    if schedule.is_enabled:
        schedule.next_run_at = _compute_next_run(schedule.cron_expression, schedule.timezone)
    else:
        schedule.next_run_at = None

    await session.commit()
    await session.refresh(schedule)
    return _to_response(schedule)


@router.delete("/workflow-schedules/{schedule_id}", status_code=204)
async def delete_schedule(
    schedule_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Delete a workflow schedule."""
    schedule = await _get_schedule_or_404(session, schedule_id, user)
    await session.delete(schedule)
    await session.commit()
