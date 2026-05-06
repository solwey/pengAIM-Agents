"""DB-backed permission tests for the workflows router.

Workflows scope by `(user_id, team_id)` like threads. Tests cover the
list/get/update/delete/restore paths plus the role matrix from
``ROLE_PERMS``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.api import workflows as workflows_module
from aegra_api.core.orm import Workflow as WorkflowORM
from tests.fixtures.permissions import (
    ROLE_PERMS,
    SCOPE_TEAM_ID,
    SCOPE_USER_ID,
    http_client,
    make_db_app_builder,
    perm,
    tenant_url,
)


@pytest.fixture
def make_app(
    project_auth: Any,
    db_conn: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., FastAPI]:
    return make_db_app_builder(
        project_auth=project_auth,
        db_conn=db_conn,
        routers=[workflows_module.router],
        monkeypatch=monkeypatch,
    )


_VALID_DEFINITION: dict[str, Any] = {"name": "wf", "nodes": [], "edges": []}


async def _seed_workflow(
    session: AsyncSession,
    workflow_id: str,
    *,
    user_id: str,
    team_id: str,
    deleted: bool = False,
) -> None:
    from datetime import UTC, datetime

    session.add(
        WorkflowORM(
            id=workflow_id,
            team_id=team_id,
            user_id=user_id,
            name=workflow_id,
            description=None,
            definition=_VALID_DEFINITION,
            deleted_at=datetime.now(UTC) if deleted else None,
        )
    )
    await session.commit()


async def _seed_three_workflows(session: AsyncSession) -> None:
    await _seed_workflow(session, "wf-own", user_id=SCOPE_USER_ID, team_id=SCOPE_TEAM_ID)
    await _seed_workflow(session, "wf-team", user_id="other-user", team_id=SCOPE_TEAM_ID)
    await _seed_workflow(session, "wf-other", user_id="other-user", team_id="other-team")


def _ids_from_list(payload: dict[str, Any]) -> set[str]:
    return {w["id"] for w in payload["items"]}


# --- Per-reach: list_workflows --------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reach", "expected"),
    [
        ("own", {"wf-own"}),
        ("team", {"wf-own", "wf-team"}),
        ("all", {"wf-own", "wf-team", "wf-other"}),
    ],
)
async def test_list_workflows_per_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    reach: str,
    expected: set[str],
) -> None:
    await _seed_three_workflows(db_session)
    app = make_app([perm("workflows", "read", reach)])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/workflows"))
    assert resp.status_code == 200, resp.text
    assert _ids_from_list(resp.json()) == expected


@pytest.mark.asyncio
async def test_list_workflows_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_workflows(db_session)
    app = make_app([perm("threads", "read", "all")])  # unrelated perm
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/workflows"))
    assert resp.status_code == 403


# --- Per-reach: get_workflow ----------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reach", "wf_id", "expected_status"),
    [
        ("own", "wf-team", 404),
        ("team", "wf-team", 200),
        ("team", "wf-other", 404),
        ("all", "wf-other", 200),
    ],
)
async def test_get_workflow_per_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    reach: str,
    wf_id: str,
    expected_status: int,
) -> None:
    await _seed_three_workflows(db_session)
    app = make_app([perm("workflows", "read", reach)])
    async with http_client(app) as client:
        resp = await client.get(tenant_url(f"/workflows/{wf_id}"))
    assert resp.status_code == expected_status, resp.text


# --- Per-reach: delete_workflow -------------------------------------------


@pytest.mark.asyncio
async def test_delete_workflow_other_team_is_404_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_workflows(db_session)
    app = make_app([perm("workflows", "delete", "team")])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/workflows/wf-other"))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_workflow_team_succeeds_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_workflows(db_session)
    app = make_app([perm("workflows", "delete", "team")])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/workflows/wf-team"))
    assert resp.status_code == 204, resp.text


@pytest.mark.asyncio
async def test_delete_workflow_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_workflows(db_session)
    app = make_app([perm("workflows", "read", "all")])  # no delete
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/workflows/wf-own"))
    assert resp.status_code == 403


# --- Per-reach: update_workflow -------------------------------------------


@pytest.mark.asyncio
async def test_update_workflow_team_succeeds_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_workflows(db_session)
    app = make_app([perm("workflows", "update", "team")])
    async with http_client(app) as client:
        resp = await client.put(
            tenant_url("/workflows/wf-team"), json={"description": "updated"}
        )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_update_workflow_other_team_is_404_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_workflows(db_session)
    app = make_app([perm("workflows", "update", "team")])
    async with http_client(app) as client:
        resp = await client.put(
            tenant_url("/workflows/wf-other"), json={"description": "updated"}
        )
    assert resp.status_code == 404


# --- Per-reach: restore_workflow ------------------------------------------


@pytest.mark.asyncio
async def test_restore_workflow_team_succeeds_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_workflow(
        db_session, "wf-deleted", user_id="other-user", team_id=SCOPE_TEAM_ID, deleted=True
    )
    app = make_app([perm("workflows", "update", "team")])
    async with http_client(app) as client:
        resp = await client.post(tenant_url("/workflows/wf-deleted/restore"))
    assert resp.status_code == 200, resp.text


# --- create_workflow ------------------------------------------------------


@pytest.mark.asyncio
async def test_create_workflow_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    # We only assert the auth gate fires before validation; the 403 is from
    # `_resolve_workflow_filters` raising before `_validate_definition` runs.
    app = make_app([perm("workflows", "read", "all")])
    async with http_client(app) as client:
        resp = await client.post(
            tenant_url("/workflows"),
            json={"name": "new-wf", "definition": _VALID_DEFINITION},
        )
    assert resp.status_code == 403


# --- Role-based -----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("superadmin", {"wf-own", "wf-team", "wf-other"}),
        ("admin", {"wf-own", "wf-team"}),
        ("user", {"wf-own"}),
    ],
)
async def test_role_list_workflows(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    expected: set[str],
) -> None:
    await _seed_three_workflows(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/workflows"))
    assert resp.status_code == 200, resp.text
    assert _ids_from_list(resp.json()) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "wf_id", "expected_status"),
    [
        ("superadmin", "wf-other", 200),
        ("admin", "wf-other", 404),
        ("admin", "wf-team", 200),
        ("user", "wf-team", 404),
        ("user", "wf-own", 200),
    ],
)
async def test_role_get_workflow(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    wf_id: str,
    expected_status: int,
) -> None:
    await _seed_three_workflows(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.get(tenant_url(f"/workflows/{wf_id}"))
    assert resp.status_code == expected_status, resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "wf_id", "expected_status"),
    [
        ("superadmin", "wf-other", 204),
        ("admin", "wf-other", 404),
        ("admin", "wf-team", 204),
        ("user", "wf-team", 404),
        ("user", "wf-own", 204),
    ],
)
async def test_role_delete_workflow(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    wf_id: str,
    expected_status: int,
) -> None:
    await _seed_three_workflows(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url(f"/workflows/{wf_id}"))
    assert resp.status_code == expected_status, resp.text
