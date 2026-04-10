"""CRUD endpoints for Workflow definitions."""

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.settings import settings

from ..core.auth_deps import auth_dependency, get_current_user
from ..core.orm import Tenant, Workflow, WorkflowVersion, get_session
from ..core.tenant import get_current_tenant
from ..models.auth import User
from ..services.webhook_security import generate_webhook_path, generate_webhook_secret

router = APIRouter(tags=["Workflows"], dependencies=auth_dependency)


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
    version: int = 1
    webhook_enabled: bool = False
    webhook_url: str | None = None
    webhook_path: str | None = None
    webhook_secret: str | None = None
    created_at: str
    updated_at: str
    deleted_at: str | None = None

    model_config = {"from_attributes": True}


class WorkflowListResponse(BaseModel):
    items: list[WorkflowResponse]
    total: int
    page: int
    page_size: int


class WebhookConfigResponse(BaseModel):
    enabled: bool
    webhook_url: str
    webhook_secret: str
    webhook_path: str


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


def _build_webhook_url(tenant_uuid: str, webhook_path: str) -> str:
    """Construct the full public webhook URL including tenant prefix."""
    return f"{settings.app.SERVER_URL.rstrip('/')}/tenant/{tenant_uuid}/webhook/{webhook_path}"


def _scope_workflow_query(query, user: User):
    """Apply team + owner scope (admins can access whole team)."""
    query = query.where(Workflow.team_id == user.team_id)
    if not user.is_admin:
        query = query.where(Workflow.user_id == user.id)
    return query


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/workflows", response_model=WorkflowResponse, status_code=201)
async def create_workflow(
    body: WorkflowCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
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
    return _to_response(workflow, tenant.uuid)


@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    is_active: bool | None = None,
    show_deleted: bool = False,
):
    """List workflows for the current team."""
    query = _scope_workflow_query(select(Workflow), user)

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
        items=[_to_response(w, tenant.uuid) for w in workflows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Get a single workflow by ID."""
    workflow = await _get_workflow_or_404(session, workflow_id, user)
    return _to_response(workflow, tenant.uuid)


@router.put("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    body: WorkflowUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Update a workflow. Validates the definition if provided.

    When the definition changes, a new version snapshot is created automatically.
    """
    workflow = await _get_workflow_or_404(session, workflow_id, user)

    definition_changed = False
    if body.definition is not None:
        _validate_definition(body.definition)
        if body.definition != workflow.definition:
            definition_changed = True
        workflow.definition = body.definition

    if body.name is not None:
        workflow.name = body.name
    if body.description is not None:
        workflow.description = body.description
    if body.is_active is not None:
        workflow.is_active = body.is_active

    if definition_changed:
        workflow.version += 1
        session.add(
            WorkflowVersion(
                workflow_id=workflow.id,
                version=workflow.version,
                definition=workflow.definition,
                name=workflow.name,
                description=workflow.description,
                created_by=user.id,
            )
        )

    workflow.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(workflow)
    return _to_response(workflow, tenant.uuid)


@router.post("/workflows/{workflow_id}/restore", response_model=WorkflowResponse)
async def restore_workflow(
    workflow_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Restore a soft-deleted workflow."""
    result = await session.execute(
        _scope_workflow_query(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.deleted_at.is_not(None),
            ),
            user,
        )
    )
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Deleted workflow not found")
    workflow.deleted_at = None
    workflow.updated_at = datetime.now(UTC)
    await session.commit()
    await session.refresh(workflow)
    return _to_response(workflow, tenant.uuid)


@router.delete("/workflows/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Soft-delete a workflow by setting deleted_at."""
    workflow = await _get_workflow_or_404(session, workflow_id, user)
    workflow.deleted_at = datetime.now(UTC)
    await session.commit()


# ---------------------------------------------------------------------------
# Version management endpoints
# ---------------------------------------------------------------------------


class WorkflowVersionResponse(BaseModel):
    workflow_id: str
    version: int
    name: str
    description: str | None
    definition: dict[str, Any] | None = None
    created_by: str
    created_at: str

    model_config = {"from_attributes": True}


@router.get("/workflows/{workflow_id}/versions")
async def list_workflow_versions(
    workflow_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
):
    """List all versions of a workflow."""
    await _get_workflow_or_404(session, workflow_id, user)

    result = await session.execute(
        select(WorkflowVersion)
        .where(WorkflowVersion.workflow_id == workflow_id)
        .order_by(WorkflowVersion.version.desc())
    )
    versions = result.scalars().all()

    return [
        WorkflowVersionResponse(
            workflow_id=v.workflow_id,
            version=v.version,
            name=v.name,
            description=v.description,
            definition=v.definition,
            created_by=v.created_by,
            created_at=v.created_at.isoformat(),
        )
        for v in versions
    ]


@router.post(
    "/workflows/{workflow_id}/versions/{version}/restore",
    response_model=WorkflowResponse,
)
async def restore_workflow_version(
    workflow_id: str,
    version: int,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Restore a workflow to a previous version."""
    workflow = await _get_workflow_or_404(session, workflow_id, user)

    result = await session.execute(
        select(WorkflowVersion).where(
            WorkflowVersion.workflow_id == workflow_id,
            WorkflowVersion.version == version,
        )
    )
    ver = result.scalar_one_or_none()
    if not ver:
        raise HTTPException(status_code=404, detail=f"Version {version} not found")

    workflow.definition = ver.definition
    workflow.name = ver.name
    workflow.description = ver.description
    workflow.version += 1
    workflow.updated_at = datetime.now(UTC)

    # Create a snapshot for the restore action
    session.add(
        WorkflowVersion(
            workflow_id=workflow.id,
            version=workflow.version,
            definition=workflow.definition,
            name=workflow.name,
            description=workflow.description,
            created_by=user.id,
        )
    )

    await session.commit()
    await session.refresh(workflow)
    return _to_response(workflow, tenant.uuid)


# ---------------------------------------------------------------------------
# Webhook management endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/workflows/{workflow_id}/webhook",
    response_model=WebhookConfigResponse,
)
async def enable_webhook(
    workflow_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Enable webhook trigger for a workflow.

    Generates a unique webhook URL and secret.  The secret is returned
    only in this response — store it securely.
    """
    workflow = await _get_workflow_or_404(session, workflow_id, user)

    if workflow.webhook_enabled and workflow.webhook_path:
        raise HTTPException(
            status_code=409,
            detail="Webhook already enabled. Use /webhook/regenerate to get a new secret.",
        )

    path = generate_webhook_path()
    secret = generate_webhook_secret()

    workflow.webhook_enabled = True
    workflow.webhook_path = path
    workflow.webhook_secret = secret
    workflow.updated_at = datetime.now(UTC)

    await session.commit()
    await session.refresh(workflow)

    return WebhookConfigResponse(
        enabled=True,
        webhook_url=_build_webhook_url(tenant.uuid, path),
        webhook_secret=secret,
        webhook_path=path,
    )


@router.delete("/workflows/{workflow_id}/webhook", status_code=204)
async def disable_webhook(
    workflow_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Disable webhook trigger for a workflow."""
    workflow = await _get_workflow_or_404(session, workflow_id, user)

    workflow.webhook_enabled = False
    workflow.webhook_path = None
    workflow.webhook_secret = None
    workflow.updated_at = datetime.now(UTC)

    await session.commit()


@router.post(
    "/workflows/{workflow_id}/webhook/regenerate",
    response_model=WebhookConfigResponse,
)
async def regenerate_webhook_secret(
    workflow_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
):
    """Regenerate the webhook secret.  Keeps the same URL path."""
    workflow = await _get_workflow_or_404(session, workflow_id, user)

    if not workflow.webhook_enabled or not workflow.webhook_path:
        raise HTTPException(status_code=400, detail="Webhook is not enabled")

    secret = generate_webhook_secret()
    workflow.webhook_secret = secret
    workflow.updated_at = datetime.now(UTC)

    await session.commit()
    await session.refresh(workflow)

    return WebhookConfigResponse(
        enabled=True,
        webhook_url=_build_webhook_url(tenant.uuid, workflow.webhook_path),
        webhook_secret=secret,
        webhook_path=workflow.webhook_path,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_workflow_or_404(session: AsyncSession, workflow_id: str, user: User) -> Workflow:
    result = await session.execute(
        _scope_workflow_query(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.deleted_at.is_(None),
            ),
            user,
        )
    )
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------


class WorkflowExport(BaseModel):
    """Portable workflow format for import/export."""
    name: str
    description: str | None = None
    definition: dict


class WorkflowImport(BaseModel):
    """Import a workflow from JSON."""
    name: str | None = None  # Override name, or use from JSON
    description: str | None = None
    definition: dict


@router.get("/workflows/{workflow_id}/export", response_model=WorkflowExport)
async def export_workflow(
    workflow_id: str,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Export a workflow as portable JSON (can be imported into another workspace)."""
    workflow = await _get_workflow_or_404(session, workflow_id, user.team_id)
    return WorkflowExport(
        name=workflow.name,
        description=workflow.description,
        definition=workflow.definition,
    )


@router.post("/workflows/import", response_model=WorkflowResponse, status_code=201)
async def import_workflow(
    body: WorkflowImport,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Import a workflow from JSON. Creates a new workflow in the current team."""
    _validate_definition(body.definition)

    workflow = Workflow(
        team_id=user.team_id,
        user_id=user.id,
        name=body.name or body.definition.get("name", "Imported Workflow"),
        description=body.description,
        definition=body.definition,
    )
    session.add(workflow)
    await session.commit()
    await session.refresh(workflow)
    return _to_response(workflow)




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_response(workflow: Workflow, tenant_uuid: str) -> WorkflowResponse:
    webhook_url = None
    webhook_path = None
    if workflow.webhook_enabled and workflow.webhook_path:
        webhook_url = _build_webhook_url(tenant_uuid, workflow.webhook_path)
        webhook_path = workflow.webhook_path

    return WorkflowResponse(
        id=workflow.id,
        team_id=workflow.team_id,
        user_id=workflow.user_id,
        name=workflow.name,
        description=workflow.description,
        definition=workflow.definition,
        is_active=workflow.is_active,
        version=workflow.version,
        webhook_enabled=workflow.webhook_enabled,
        webhook_url=webhook_url,
        webhook_path=webhook_path,
        webhook_secret=workflow.webhook_secret if workflow.webhook_enabled else None,
        created_at=workflow.created_at.isoformat(),
        updated_at=workflow.updated_at.isoformat(),
        deleted_at=workflow.deleted_at.isoformat() if workflow.deleted_at else None,
    )
