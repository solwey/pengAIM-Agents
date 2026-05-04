"""DB-backed permission tests for the workflow_runs router.

Workflow runs scope by `(user_id, team_id)`. ``ROLE_PERMS`` carries
own/team/all reach for each role; tests verify list/get/delete/cancel
behaviour matches the reach.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.api import workflow_runs as workflow_runs_module
from aegra_api.core.orm import Workflow as WorkflowORM
from aegra_api.core.orm import WorkflowRun as WorkflowRunORM
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
        routers=[workflow_runs_module.router],
        monkeypatch=monkeypatch,
    )


async def _seed_workflow(
    session: AsyncSession,
    workflow_id: str,
    *,
    user_id: str,
    team_id: str,
) -> None:
    session.add(
        WorkflowORM(
            id=workflow_id,
            team_id=team_id,
            user_id=user_id,
            name=workflow_id,
            definition={"name": workflow_id, "nodes": [], "edges": []},
        )
    )
    await session.commit()


async def _seed_run(
    session: AsyncSession,
    run_id: str,
    *,
    workflow_id: str,
    user_id: str,
    team_id: str,
    status: str = "completed",
) -> None:
    session.add(
        WorkflowRunORM(
            id=run_id,
            workflow_id=workflow_id,
            team_id=team_id,
            user_id=user_id,
            status=status,
        )
    )
    await session.commit()


async def _seed_three_runs(session: AsyncSession) -> None:
    """One workflow+run per scope: own / same-team / other-team."""
    await _seed_workflow(session, "wf-own", user_id=SCOPE_USER_ID, team_id=SCOPE_TEAM_ID)
    await _seed_run(
        session, "wr-own", workflow_id="wf-own", user_id=SCOPE_USER_ID, team_id=SCOPE_TEAM_ID
    )
    await _seed_workflow(session, "wf-team", user_id="other-user", team_id=SCOPE_TEAM_ID)
    await _seed_run(
        session, "wr-team", workflow_id="wf-team", user_id="other-user", team_id=SCOPE_TEAM_ID
    )
    await _seed_workflow(session, "wf-other", user_id="other-user", team_id="other-team")
    await _seed_run(
        session, "wr-other", workflow_id="wf-other", user_id="other-user", team_id="other-team"
    )


def _ids(payload: dict[str, Any]) -> set[str]:
    return {r["id"] for r in payload["items"]}


# --- Per-reach: list_workflow_runs ----------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reach", "expected"),
    [
        ("own", {"wr-own"}),
        ("team", {"wr-own", "wr-team"}),
        ("all", {"wr-own", "wr-team", "wr-other"}),
    ],
)
async def test_list_workflow_runs_per_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    reach: str,
    expected: set[str],
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("workflow_runs", "read", reach)])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/workflow-runs"))
    assert resp.status_code == 200, resp.text
    assert _ids(resp.json()) == expected


@pytest.mark.asyncio
async def test_list_workflow_runs_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("workflows", "read", "all")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/workflow-runs"))
    assert resp.status_code == 403


# --- Per-reach: get/delete ------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reach", "run_id", "expected_status"),
    [
        ("own", "wr-team", 404),
        ("team", "wr-team", 200),
        ("team", "wr-other", 404),
        ("all", "wr-other", 200),
    ],
)
async def test_get_workflow_run_per_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    reach: str,
    run_id: str,
    expected_status: int,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("workflow_runs", "read", reach)])
    async with http_client(app) as client:
        resp = await client.get(tenant_url(f"/workflow-runs/{run_id}"))
    assert resp.status_code == expected_status, resp.text


@pytest.mark.asyncio
async def test_delete_workflow_run_other_team_is_404_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("workflow_runs", "delete", "team")])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/workflow-runs/wr-other"))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_workflow_run_team_succeeds_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("workflow_runs", "delete", "team")])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/workflow-runs/wr-team"))
    assert resp.status_code == 204, resp.text


@pytest.mark.asyncio
async def test_cancel_workflow_run_team_succeeds_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    """Seed a `running` row so the cancel endpoint can transition it."""
    await _seed_workflow(
        db_session, "wf-cancel", user_id="other-user", team_id=SCOPE_TEAM_ID
    )
    await _seed_run(
        db_session,
        "wr-running",
        workflow_id="wf-cancel",
        user_id="other-user",
        team_id=SCOPE_TEAM_ID,
        status="running",
    )
    app = make_app([perm("workflow_runs", "update", "team")])
    async with http_client(app) as client:
        resp = await client.post(tenant_url("/workflow-runs/wr-running/cancel"))
    assert resp.status_code == 200, resp.text


# --- Role-based -----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("superadmin", {"wr-own", "wr-team", "wr-other"}),
        ("admin", {"wr-own", "wr-team"}),
        ("user", {"wr-own"}),
    ],
)
async def test_role_list_workflow_runs(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    expected: set[str],
) -> None:
    await _seed_three_runs(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/workflow-runs"))
    assert resp.status_code == 200, resp.text
    assert _ids(resp.json()) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "run_id", "expected_status"),
    [
        ("superadmin", "wr-other", 200),
        ("admin", "wr-other", 404),
        ("admin", "wr-team", 200),
        ("user", "wr-team", 404),
        ("user", "wr-own", 200),
    ],
)
async def test_role_get_workflow_run(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    run_id: str,
    expected_status: int,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.get(tenant_url(f"/workflow-runs/{run_id}"))
    assert resp.status_code == expected_status, resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "run_id", "expected_status"),
    [
        ("superadmin", "wr-other", 204),
        ("admin", "wr-other", 404),
        ("admin", "wr-team", 204),
        ("user", "wr-team", 404),
        ("user", "wr-own", 204),
    ],
)
async def test_role_delete_workflow_run(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    run_id: str,
    expected_status: int,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url(f"/workflow-runs/{run_id}"))
    assert resp.status_code == expected_status, resp.text
