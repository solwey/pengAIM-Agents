"""DB-backed permission tests for the control_plane router.

Control-plane endpoints gate on `control_plane.read` (overview/list) or
`control_plane.cleanup` (worker cleanup). The active-runs / stats blocks
of `/overview` are scoped by `team_id` (own and team reach both yield
`{"team_id": ...}`); `.all` reach drops the filter and returns rows
across teams. The `/workers` and `/celery-workers` listings are not
scoped — they only require the read perm.

`USER_PERMS` carries no `control_plane.*` permissions, so users get 403
on every endpoint.

Celery is monkey-patched to return no workers — the real inspector would
make a network call with a 1-second timeout.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.api import control_plane as control_plane_module
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
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
    # Skip the live Celery inspect call; otherwise `/overview` and
    # `/celery-workers` would block on a 1-second network timeout per call.
    async def _no_celery_workers() -> list[Any]:
        return []

    monkeypatch.setattr(
        control_plane_module, "_get_celery_workers", _no_celery_workers
    )

    return make_db_app_builder(
        project_auth=project_auth,
        db_conn=db_conn,
        routers=[control_plane_module.router],
        monkeypatch=monkeypatch,
    )


# --- Seed helpers ----------------------------------------------------------


async def _seed_thread(
    session: AsyncSession,
    thread_id: str,
    *,
    user_id: str,
    team_id: str,
) -> None:
    session.add(
        ThreadORM(
            thread_id=thread_id,
            status="idle",
            metadata_json={},
            user_id=user_id,
            team_id=team_id,
        )
    )
    await session.commit()


async def _seed_running_run(
    session: AsyncSession,
    run_id: str,
    *,
    thread_id: str,
    user_id: str,
    team_id: str,
) -> None:
    session.add(
        RunORM(
            run_id=run_id,
            thread_id=thread_id,
            status="running",
            user_id=user_id,
            team_id=team_id,
        )
    )
    await session.commit()


async def _seed_two_active_runs(session: AsyncSession) -> None:
    """One running run in caller's team, one in another team."""
    await _seed_thread(session, "t-own", user_id=SCOPE_USER_ID, team_id=SCOPE_TEAM_ID)
    await _seed_running_run(
        session, "r-own", thread_id="t-own", user_id=SCOPE_USER_ID, team_id=SCOPE_TEAM_ID
    )
    await _seed_thread(session, "t-other", user_id="other-user", team_id="other-team")
    await _seed_running_run(
        session, "r-other", thread_id="t-other", user_id="other-user", team_id="other-team"
    )


# --- /overview -------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reach", "expected_active_ids"),
    [
        # `team` reach scopes active_runs to caller's team_id.
        ("team", {"r-own"}),
        # `all` reach drops the filter — returns runs across teams.
        ("all", {"r-own", "r-other"}),
    ],
)
async def test_overview_scopes_active_runs_by_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    reach: str,
    expected_active_ids: set[str],
) -> None:
    await _seed_two_active_runs(db_session)
    app = make_app([perm("control_plane", "read", reach)])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/control-plane/overview"))
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert {r["run_id"] for r in payload["active_runs"]} == expected_active_ids


@pytest.mark.asyncio
async def test_overview_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app([perm("threads", "read", "all")])  # unrelated perm
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/control-plane/overview"))
    assert resp.status_code == 403


# --- /workers --------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("reach", ["team", "all"])
async def test_list_workers_succeeds_with_read_perm(
    make_app: Callable[..., FastAPI],
    reach: str,
) -> None:
    """`/workers` doesn't scope by team — any read reach passes the auth gate."""
    app = make_app([perm("control_plane", "read", reach)])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/control-plane/workers"))
    assert resp.status_code == 200, resp.text
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_list_workers_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app([perm("threads", "read", "all")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/control-plane/workers"))
    assert resp.status_code == 403


# --- /celery-workers -------------------------------------------------------


@pytest.mark.asyncio
async def test_list_celery_workers_succeeds_with_read_perm(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app([perm("control_plane", "read", "team")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/control-plane/celery-workers"))
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_list_celery_workers_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app([perm("control_plane", "cleanup", "all")])  # has cleanup, no read
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/control-plane/celery-workers"))
    assert resp.status_code == 403


# --- DELETE /workers/cleanup ----------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("reach", ["team", "all"])
async def test_cleanup_workers_succeeds_with_cleanup_perm(
    make_app: Callable[..., FastAPI],
    reach: str,
) -> None:
    app = make_app([perm("control_plane", "cleanup", reach)])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/control-plane/workers/cleanup"))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"removed": 0}


@pytest.mark.asyncio
async def test_cleanup_workers_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app([perm("control_plane", "read", "all")])  # has read, no cleanup
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/control-plane/workers/cleanup"))
    assert resp.status_code == 403


# --- Role-based -----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "expected_status", "expected_active_ids"),
    [
        ("superadmin", 200, {"r-own", "r-other"}),
        ("admin", 200, {"r-own"}),
        # `user` role has no control_plane permissions at all.
        ("user", 403, None),
    ],
)
async def test_role_overview(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    expected_status: int,
    expected_active_ids: set[str] | None,
) -> None:
    await _seed_two_active_runs(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/control-plane/overview"))
    assert resp.status_code == expected_status, resp.text
    if expected_active_ids is not None:
        ids = {r["run_id"] for r in resp.json()["active_runs"]}
        assert ids == expected_active_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "expected_status"),
    [
        ("superadmin", 200),
        ("admin", 200),
        ("user", 403),
    ],
)
async def test_role_cleanup_workers(
    make_app: Callable[..., FastAPI],
    role: str,
    expected_status: int,
) -> None:
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/control-plane/workers/cleanup"))
    assert resp.status_code == expected_status, resp.text
