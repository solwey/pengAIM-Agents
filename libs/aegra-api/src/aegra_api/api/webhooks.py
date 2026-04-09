"""Public webhook endpoint for triggering workflow runs.

This endpoint is mounted outside the auth middleware so it does not
require JWT authentication.  Instead it verifies requests using either
HMAC-SHA256 signatures or Bearer tokens.
"""

import json
import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Request
from jose import jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.settings import settings

from ..core.orm import Workflow, WorkflowRun, get_session, get_current_tenant, Tenant
from ..services.webhook_security import (
    WebhookVerificationError,
    verify_bearer_token,
    verify_hmac_signature,
)

logger = logging.getLogger("webhooks")

router = APIRouter(tags=["Webhooks"])


def _decode_jwt_key(v: str | None) -> str | None:
    """Decode base64-encoded key to PEM, or return as-is if already PEM."""
    if not v:
        return None
    v = v.strip()
    if v.startswith("-----BEGIN"):
        return v
    try:
        import base64

        return base64.b64decode(v).decode("utf-8")
    except Exception:
        return v


def _create_internal_token(user_id: str, team_id: str, tenant_id: str) -> str:
    """Create a short-lived JWT for internal service calls on behalf of
    the workflow owner.  Uses the same signing key / algorithm that
    pengAIM-RAG uses so the revops-backend accepts it transparently.
    """
    private_key = _decode_jwt_key(settings.app.JWT_PRIVATE_KEY)

    if not private_key:
        logger.warning("[webhooks] No JWT_PRIVATE_KEY or JWT_SECRET — cannot mint internal token")
        return ""

    now = datetime.now(UTC)
    claims = {
        "aud": tenant_id,
        "sub": user_id,
        "team_id": team_id,
        "role": "owner",
        "typ": "access",
        "iat": now,
        "exp": now + timedelta(minutes=30),
    }
    return jwt.encode(claims, private_key, algorithm=settings.app.JWT_ALG)


@router.post("/{webhook_path}")
async def handle_workflow_webhook(
    webhook_path: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
    tenant: Tenant = Depends(get_current_tenant),
) -> dict[str, str]:
    """Receive an incoming webhook and trigger the associated workflow run."""
    # 1. Look up workflow by webhook_path
    result = await session.execute(
        select(Workflow).where(
            Workflow.webhook_path == webhook_path,
            Workflow.webhook_enabled.is_(True),
            Workflow.is_active.is_(True),
            Workflow.deleted_at.is_(None),
        )
    )
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Webhook not found")

    # 2. Read raw body
    body = await request.body()

    # 3. Verify authentication (optional — some services like RB2B
    #    send plain POST without any auth headers)
    signature = request.headers.get("x-webhook-signature")
    timestamp = request.headers.get("x-webhook-timestamp")
    auth_header = request.headers.get("authorization", "")

    if signature and timestamp and workflow.webhook_secret:
        try:
            verify_hmac_signature(workflow.webhook_secret, body, signature, timestamp)
        except WebhookVerificationError as e:
            logger.warning(f"[webhooks] HMAC verification failed for path={webhook_path}: {e}")
            raise HTTPException(status_code=401, detail="Unauthorized")
    elif auth_header.lower().startswith("bearer ") and workflow.webhook_secret:
        try:
            token = auth_header[7:]
            verify_bearer_token(workflow.webhook_secret, token)
        except WebhookVerificationError as e:
            logger.warning(f"[webhooks] Bearer verification failed for path={webhook_path}: {e}")
            raise HTTPException(status_code=401, detail="Unauthorized")

    # 4. Parse body as JSON
    try:
        body_json = json.loads(body) if body else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body_json, dict):
        body_json = {"payload": body_json}

    # Normalize keys: "First Name" → "first_name" (for RB2B and similar)
    normalized = {}
    for key, value in body_json.items():
        norm_key = key.strip().lower().replace(" ", "_")
        normalized[norm_key] = value

    input_data = {
        **normalized,
        "_webhook": {
            "received_at": datetime.now(UTC).isoformat(),
            "source_ip": request.client.host if request.client else None,
            "raw_keys": list(body_json.keys()),
        },
    }

    # 5. Create workflow run
    run = WorkflowRun(
        workflow_id=workflow.id,
        team_id=workflow.team_id,
        user_id="webhook",
        status="pending",
        input_data=input_data,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    # 6. Generate internal auth token for the workflow owner so that
    #    downstream nodes (create_account, create_contact, etc.) can
    #    call the revops-backend on behalf of the workflow creator.
    internal_token = _create_internal_token(workflow.user_id, workflow.team_id, tenant.uuid)
    auth_token = f"Bearer {internal_token}" if internal_token else ""

    # 7. Queue Celery task
    from ..tasks import execute_workflow

    task = execute_workflow.delay(run.id, auth_token=auth_token)
    run.celery_task_id = task.id
    await session.commit()

    logger.info(f"[webhooks] Workflow run created: run_id={run.id}, workflow_id={workflow.id}, path={webhook_path}")

    return {"run_id": run.id, "status": "pending"}
