"""Workflow node preview endpoints — single-node dry-runs without WorkflowRun persistence.

Used by the workflow editor to let users test a node's config against a sample input
before saving the full workflow.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
from graphs.workflow_engine.nodes.icp_score import score_account
from graphs.workflow_engine.schema import ICPScoreConfig
from pydantic import BaseModel, Field

from ..core.auth_deps import auth_dependency, get_current_user
from ..models.auth import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Workflow Node Preview"], dependencies=auth_dependency)


class ICPScorePreviewRequest(BaseModel):
    account_data: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class ICPScorePreviewResponse(BaseModel):
    ok: bool
    score: int | None = None
    status: str | None = None
    reasoning: str | None = None
    error: str | None = None


@router.post("/workflow-nodes/icp-score/preview", response_model=ICPScorePreviewResponse)
async def preview_icp_score(
    body: ICPScorePreviewRequest,
    request: Request,
    user: User = Depends(get_current_user),
) -> ICPScorePreviewResponse:
    """Run the ICP score node on a single account without persisting a WorkflowRun."""
    if not body.account_data:
        return ICPScorePreviewResponse(ok=False, error="account_data is required")

    cfg = ICPScoreConfig(**body.config)
    auth_token = request.headers.get("authorization", "")

    result = await score_account(
        body.account_data,
        model=cfg.model,
        hot_threshold=cfg.hot_threshold,
        warm_threshold=cfg.warm_threshold,
        custom_criteria=cfg.custom_criteria,
        auth_token=auth_token,
    )

    return ICPScorePreviewResponse(
        ok=bool(result.get("ok")),
        score=result.get("score"),
        status=result.get("status"),
        reasoning=result.get("reasoning"),
        error=result.get("error"),
    )
