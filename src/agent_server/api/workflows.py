"""CRUD endpoints for Workflow definitions."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.auth_deps import get_current_user
from ..core.orm import Workflow, get_session
from ..models.auth import User

router = APIRouter(tags=["Workflows"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class WorkflowCreate(BaseModel):
    name: str
    description: str | None = None
    definition: dict = Field(..., description="Workflow JSON (nodes + edges)")


class WorkflowUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    definition: dict | None = None
    is_active: bool | None = None


class WorkflowResponse(BaseModel):
    id: str
    team_id: str
    user_id: str
    name: str
    description: str | None
    definition: dict
    is_active: bool
    created_at: str
    updated_at: str
    deleted_at: str | None = None

    model_config = {"from_attributes": True}


class WorkflowListResponse(BaseModel):
    items: list[WorkflowResponse]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_definition(definition: dict) -> None:
    """Validate workflow definition using the Pydantic schema.

    Raises HTTPException 422 if invalid.
    """
    from graphs.workflow_engine.schema import WorkflowDefinition

    try:
        WorkflowDefinition(**definition)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid workflow definition: {e}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/workflows", response_model=WorkflowResponse, status_code=201)
async def create_workflow(
    body: WorkflowCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Create a new workflow. Validates the definition on creation."""
    _validate_definition(body.definition)

    workflow = Workflow(
        team_id=user.team_id,
        user_id=user.id,
        name=body.name,
        description=body.description,
        definition=body.definition,
    )
    session.add(workflow)
    await session.commit()
    await session.refresh(workflow)
    return _to_response(workflow)


@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    is_active: bool | None = None,
    show_deleted: bool = False,
):
    """List workflows for the current team."""
    query = select(Workflow).where(Workflow.team_id == user.team_id)

    if not show_deleted:
        query = query.where(Workflow.deleted_at.is_(None))

    if is_active is not None:
        query = query.where(Workflow.is_active == is_active)

    # Count
    count_query = select(func.count()).select_from(query.subquery())
    total = (await session.execute(count_query)).scalar() or 0

    # Paginate
    query = query.order_by(Workflow.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await session.execute(query)
    workflows = result.scalars().all()

    return WorkflowListResponse(
        items=[_to_response(w) for w in workflows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Get a single workflow by ID."""
    workflow = await _get_workflow_or_404(session, workflow_id, user.team_id)
    return _to_response(workflow)


@router.put("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    body: WorkflowUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Update a workflow. Validates the definition if provided."""
    workflow = await _get_workflow_or_404(session, workflow_id, user.team_id)

    if body.definition is not None:
        _validate_definition(body.definition)
        workflow.definition = body.definition

    if body.name is not None:
        workflow.name = body.name
    if body.description is not None:
        workflow.description = body.description
    if body.is_active is not None:
        workflow.is_active = body.is_active

    workflow.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(workflow)
    return _to_response(workflow)


@router.post("/workflows/{workflow_id}/restore", response_model=WorkflowResponse)
async def restore_workflow(
    workflow_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Restore a soft-deleted workflow."""
    result = await session.execute(
        select(Workflow).where(
            Workflow.id == workflow_id,
            Workflow.team_id == user.team_id,
            Workflow.deleted_at.is_not(None),
        )
    )
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Deleted workflow not found")
    workflow.deleted_at = None
    workflow.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(workflow)
    return _to_response(workflow)


@router.delete("/workflows/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Soft-delete a workflow by setting deleted_at."""
    workflow = await _get_workflow_or_404(session, workflow_id, user.team_id)
    workflow.deleted_at = datetime.now(UTC)
    await session.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_workflow_or_404(
    session: AsyncSession, workflow_id: str, team_id: str
) -> Workflow:
    result = await session.execute(
        select(Workflow).where(
            Workflow.id == workflow_id,
            Workflow.team_id == team_id,
            Workflow.deleted_at.is_(None),
        )
    )
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


def _to_response(workflow: Workflow) -> WorkflowResponse:
    return WorkflowResponse(
        id=workflow.id,
        team_id=workflow.team_id,
        user_id=workflow.user_id,
        name=workflow.name,
        description=workflow.description,
        definition=workflow.definition,
        is_active=workflow.is_active,
        created_at=workflow.created_at.isoformat(),
        updated_at=workflow.updated_at.isoformat(),
        deleted_at=workflow.deleted_at.isoformat() if workflow.deleted_at else None,
    )
