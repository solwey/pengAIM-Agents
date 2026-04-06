"""Endpoints for executing and managing Workflow Runs."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.auth_deps import auth_dependency, get_current_user
from ..core.orm import Workflow, WorkflowRun, get_session
from ..models.auth import User

router = APIRouter(tags=["Workflow Runs"], dependencies=auth_dependency)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class WorkflowRunCreate(BaseModel):
    workflow_id: str
    input_data: dict | None = None


class WorkflowRunResponse(BaseModel):
    id: str
    workflow_id: str
    team_id: str
    user_id: str
    status: str
    input_data: dict | None
    output_data: dict | None
    error_message: str | None
    created_at: str
    started_at: str | None
    completed_at: str | None

    model_config = {"from_attributes": True}


class WorkflowRunListResponse(BaseModel):
    items: list[WorkflowRunResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/workflow-runs", response_model=WorkflowRunResponse, status_code=201)
async def create_workflow_run(
    body: WorkflowRunCreate,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Create a workflow run and queue it for execution via Celery."""
    # Verify workflow exists and belongs to team
    result = await session.execute(
        select(Workflow).where(
            Workflow.id == body.workflow_id,
            Workflow.team_id == user.team_id,
        )
    )
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    if not workflow.is_active:
        raise HTTPException(status_code=400, detail="Workflow is not active")

    # Create run record
    run = WorkflowRun(
        workflow_id=workflow.id,
        team_id=user.team_id,
        user_id=user.id,
        status="pending",
        input_data=body.input_data or {},
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    # Queue Celery task with auth token for api_keys resolution
    from ..tasks import execute_workflow

    auth_token = request.headers.get("authorization", "")
    task = execute_workflow.delay(run.id, auth_token=auth_token)
    run.celery_task_id = task.id
    await session.commit()

    return _to_response(run)


@router.get("/workflow-runs", response_model=WorkflowRunListResponse)
async def list_workflow_runs(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    workflow_id: str | None = None,
    status: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
):
    """List workflow runs for the current team."""
    query = select(WorkflowRun).where(WorkflowRun.team_id == user.team_id)

    if workflow_id:
        query = query.where(WorkflowRun.workflow_id == workflow_id)
    if status:
        query = query.where(WorkflowRun.status == status)

    # Count
    count_query = select(func.count()).select_from(query.subquery())
    total = (await session.execute(count_query)).scalar() or 0

    # Paginate
    query = query.order_by(WorkflowRun.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(query)
    runs = result.scalars().all()

    return WorkflowRunListResponse(
        items=[_to_response(r) for r in runs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/workflow-runs/{run_id}", response_model=WorkflowRunResponse)
async def get_workflow_run(
    run_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Get a single workflow run by ID (used for polling)."""
    run = await _get_run_or_404(session, run_id, user.team_id)
    return _to_response(run)


@router.post("/workflow-runs/{run_id}/cancel", response_model=WorkflowRunResponse)
async def cancel_workflow_run(
    run_id: str,
    force: bool = Query(False),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Cancel a running workflow run.

    Sets status to 'cancelled' in DB. The Celery task polls this field
    and cancels itself gracefully. Use force=true for emergency termination.
    """
    run = await _get_run_or_404(session, run_id, user.team_id)

    if run.status not in ("pending", "running"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel run with status '{run.status}'",
        )

    run.status = "cancelled"
    await session.commit()

    # Force terminate the Celery worker process (emergency only)
    if force and run.celery_task_id:
        from ..celery_app import celery_app

        celery_app.control.revoke(run.celery_task_id, terminate=True)

    await session.refresh(run)
    return _to_response(run)


@router.delete("/workflow-runs/{run_id}", status_code=204)
async def delete_workflow_run(
    run_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Delete a workflow run record."""
    run = await _get_run_or_404(session, run_id, user.team_id)
    await session.delete(run)
    await session.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_run_or_404(session: AsyncSession, run_id: str, team_id: str) -> WorkflowRun:
    result = await session.execute(
        select(WorkflowRun).where(
            WorkflowRun.id == run_id,
            WorkflowRun.team_id == team_id,
        )
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    return run


def _to_response(run: WorkflowRun) -> WorkflowRunResponse:
    return WorkflowRunResponse(
        id=run.id,
        workflow_id=run.workflow_id,
        team_id=run.team_id,
        user_id=run.user_id,
        status=run.status,
        input_data=run.input_data,
        output_data=run.output_data,
        error_message=run.error_message,
        created_at=run.created_at.isoformat(),
        started_at=run.started_at.isoformat() if run.started_at else None,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
    )
