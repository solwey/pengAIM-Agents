"""Permission tests for the workflow_node_preview router.

Each preview endpoint gates on `workflows.preview` via
`_check_preview_permission`. The actual preview body calls out to LLMs /
HTTP / SMTP — that's out of scope here. We assert that:
  * with the perm + an empty body, the route gets past auth and returns
    200 with `ok=False` (the body's own validation kicks in);
  * without the perm, the auth gate returns 403 before any side effects.

All three roles in `ROLE_PERMS` carry `workflows.preview`, so the role
matrix is just "every role is allowed".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI

from aegra_api.api import workflow_node_preview as preview_module
from tests.fixtures.permissions import (
    ROLE_PERMS,
    http_client,
    make_db_app_builder,
    tenant_url,
)


# Endpoints share `_check_preview_permission`. ICP score is the cheapest
# path: with empty `account_data` it short-circuits to `ok=False` before
# touching any LLM. Use it as the canonical preview endpoint for these tests.
_ENDPOINT: str = "/workflow-nodes/icp-score/preview"


@pytest.fixture
def make_app(
    project_auth: Any,
    db_conn: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., FastAPI]:
    return make_db_app_builder(
        project_auth=project_auth,
        db_conn=db_conn,
        routers=[preview_module.router],
        monkeypatch=monkeypatch,
    )


@pytest.mark.asyncio
async def test_preview_with_workflows_preview_perm_passes_auth(
    make_app: Callable[..., FastAPI],
) -> None:
    # `workflows.preview` is a no-reach permission: just the bare string.
    app = make_app(["workflows.preview"])
    async with http_client(app) as client:
        resp = await client.post(tenant_url(_ENDPOINT), json={"account_data": {}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["ok"] is False  # validation error from empty body


@pytest.mark.asyncio
async def test_preview_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app(["workflows.read.all"])  # has read but not preview
    async with http_client(app) as client:
        resp = await client.post(tenant_url(_ENDPOINT), json={"account_data": {}})
    assert resp.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["superadmin", "admin", "user"])
async def test_role_preview_allowed(
    make_app: Callable[..., FastAPI],
    role: str,
) -> None:
    """Every role in `ROLE_PERMS` carries `workflows.preview`."""
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.post(tenant_url(_ENDPOINT), json={"account_data": {}})
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    [
        "/workflow-nodes/icp-score/preview",
        "/workflow-nodes/api-request/preview",
        "/workflow-nodes/slack-message/preview",
        "/workflow-nodes/email-message/preview",
        "/workflow-nodes/llm-complete/preview",
    ],
)
async def test_every_preview_endpoint_enforces_workflows_preview(
    make_app: Callable[..., FastAPI],
    endpoint: str,
) -> None:
    """All five preview endpoints must reject callers lacking `workflows.preview`."""
    app = make_app(["workflows.read.all"])
    async with http_client(app) as client:
        resp = await client.post(tenant_url(endpoint), json={"config": {}})
    assert resp.status_code == 403
