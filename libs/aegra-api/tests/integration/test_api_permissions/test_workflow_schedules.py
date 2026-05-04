"""DB-backed permission tests for the workflow_schedules router."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.api import workflow_schedules as workflow_schedules_module
from aegra_api.core.orm import Workflow as WorkflowORM
from aegra_api.core.orm import WorkflowSchedule as WorkflowScheduleORM
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
        routers=[workflow_schedules_module.router],
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


async def _seed_schedule(
    session: AsyncSession,
    schedule_id: str,
    *,
    workflow_id: str,
    user_id: str,
    team_id: str,
    is_enabled: bool = False,
) -> None:
    # `is_enabled=False` avoids the partial unique-index collision when
    # multiple schedules share a workflow_id.
    session.add(
        WorkflowScheduleORM(
            id=schedule_id,
            workflow_id=workflow_id,
            team_id=team_id,
            user_id=user_id,
            cron_expression="0 * * * *",
            timezone="UTC",
            is_enabled=is_enabled,
        )
    )
    await session.commit()


async def _seed_three_schedules(session: AsyncSession) -> None:
    await _seed_workflow(session, "wf-own", user_id=SCOPE_USER_ID, team_id=SCOPE_TEAM_ID)
    await _seed_schedule(
        session, "sch-own", workflow_id="wf-own", user_id=SCOPE_USER_ID, team_id=SCOPE_TEAM_ID
    )
    await _seed_workflow(session, "wf-team", user_id="other-user", team_id=SCOPE_TEAM_ID)
    await _seed_schedule(
        session, "sch-team", workflow_id="wf-team", user_id="other-user", team_id=SCOPE_TEAM_ID
    )
    await _seed_workflow(session, "wf-other", user_id="other-user", team_id="other-team")
    await _seed_schedule(
        session,
        "sch-other",
        workflow_id="wf-other",
        user_id="other-user",
        team_id="other-team",
    )


def _ids(payload: list[dict[str, Any]]) -> set[str]:
    return {s["id"] for s in payload}


# --- Per-reach: list_schedules --------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reach", "expected"),
    [
        ("own", {"sch-own"}),
        ("team", {"sch-own", "sch-team"}),
        ("all", {"sch-own", "sch-team", "sch-other"}),
    ],
)
async def test_list_schedules_per_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    reach: str,
    expected: set[str],
) -> None:
    await _seed_three_schedules(db_session)
    app = make_app([perm("workflow_schedules", "read", reach)])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/workflow-schedules"))
    assert resp.status_code == 200, resp.text
    assert _ids(resp.json()) == expected


@pytest.mark.asyncio
async def test_list_schedules_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_schedules(db_session)
    app = make_app([perm("workflows", "read", "all")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/workflow-schedules"))
    assert resp.status_code == 403


# --- Per-reach: get / delete / update -------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reach", "schedule_id", "expected_status"),
    [
        ("own", "sch-team", 404),
        ("team", "sch-team", 200),
        ("team", "sch-other", 404),
        ("all", "sch-other", 200),
    ],
)
async def test_get_schedule_per_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    reach: str,
    schedule_id: str,
    expected_status: int,
) -> None:
    await _seed_three_schedules(db_session)
    app = make_app([perm("workflow_schedules", "read", reach)])
    async with http_client(app) as client:
        resp = await client.get(tenant_url(f"/workflow-schedules/{schedule_id}"))
    assert resp.status_code == expected_status, resp.text


@pytest.mark.asyncio
async def test_delete_schedule_other_team_is_404_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_schedules(db_session)
    app = make_app([perm("workflow_schedules", "delete", "team")])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/workflow-schedules/sch-other"))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_schedule_team_succeeds_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_schedules(db_session)
    app = make_app([perm("workflow_schedules", "delete", "team")])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/workflow-schedules/sch-team"))
    assert resp.status_code == 204, resp.text


@pytest.mark.asyncio
async def test_update_schedule_team_succeeds_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_schedules(db_session)
    app = make_app([perm("workflow_schedules", "update", "team")])
    async with http_client(app) as client:
        resp = await client.put(
            tenant_url("/workflow-schedules/sch-team"),
            json={"is_enabled": False},
        )
    assert resp.status_code == 200, resp.text


# --- Role-based -----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "expected"),
    [
        ("superadmin", {"sch-own", "sch-team", "sch-other"}),
        ("admin", {"sch-own", "sch-team"}),
        ("user", {"sch-own"}),
    ],
)
async def test_role_list_schedules(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    expected: set[str],
) -> None:
    await _seed_three_schedules(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/workflow-schedules"))
    assert resp.status_code == 200, resp.text
    assert _ids(resp.json()) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "schedule_id", "expected_status"),
    [
        ("superadmin", "sch-other", 200),
        ("admin", "sch-other", 404),
        ("admin", "sch-team", 200),
        ("user", "sch-team", 404),
        ("user", "sch-own", 200),
    ],
)
async def test_role_get_schedule(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    schedule_id: str,
    expected_status: int,
) -> None:
    await _seed_three_schedules(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.get(tenant_url(f"/workflow-schedules/{schedule_id}"))
    assert resp.status_code == expected_status, resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "schedule_id", "expected_status"),
    [
        ("superadmin", "sch-other", 204),
        ("admin", "sch-other", 404),
        ("admin", "sch-team", 204),
        ("user", "sch-team", 404),
        ("user", "sch-own", 204),
    ],
)
async def test_role_delete_schedule(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    schedule_id: str,
    expected_status: int,
) -> None:
    await _seed_three_schedules(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url(f"/workflow-schedules/{schedule_id}"))
    assert resp.status_code == expected_status, resp.text
