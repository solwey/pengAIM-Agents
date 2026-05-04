"""DB-backed permission tests for the threads router.

Verifies that the WHERE clauses generated from the auth filter dict
return the right rows from a real Postgres test database. Auth and
tenant-lookup deps are stubbed by ``permissions_db.make_db_app_builder``;
only the DB layer is real.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.api import threads as threads_module
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
    return make_db_app_builder(
        project_auth=project_auth,
        db_conn=db_conn,
        routers=[threads_module.router],
        monkeypatch=monkeypatch,
    )


# --- Seed helpers ----------------------------------------------------------


async def _seed_thread(
    session: AsyncSession,
    thread_id: str,
    *,
    user_id: str,
    team_id: str,
    is_shared: bool = False,
) -> None:
    session.add(
        ThreadORM(
            thread_id=thread_id,
            status="idle",
            metadata_json={"owner": user_id},
            user_id=user_id,
            team_id=team_id,
            is_shared=is_shared,
        )
    )
    await session.commit()


async def _seed_three_threads(session: AsyncSession) -> None:
    await _seed_thread(session, "own", user_id=SCOPE_USER_ID, team_id=SCOPE_TEAM_ID)
    await _seed_thread(session, "same-team", user_id="other-user", team_id=SCOPE_TEAM_ID)
    await _seed_thread(session, "other-team", user_id="other-user", team_id="other-team")


def _list_url() -> str:
    return tenant_url("/threads")


def _ids(payload: dict[str, Any]) -> set[str]:
    return {t["thread_id"] for t in payload["threads"]}


# --- Tests -----------------------------------------------------------------
# Caller is (SCOPE_USER_ID, SCOPE_TEAM_ID).


@pytest.mark.asyncio
async def test_list_own_returns_only_callers_threads(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_threads(db_session)
    app = make_app([perm("threads", "read", "own")])
    async with http_client(app) as client:
        resp = await client.request("GET", _list_url(), json={})
    assert resp.status_code == 200, resp.text
    assert _ids(resp.json()) == {"own"}


@pytest.mark.asyncio
async def test_list_team_returns_team_threads_only(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_threads(db_session)
    app = make_app([perm("threads", "read", "team")])
    async with http_client(app) as client:
        resp = await client.request("GET", _list_url(), json={})
    assert resp.status_code == 200, resp.text
    assert _ids(resp.json()) == {"own", "same-team"}


@pytest.mark.asyncio
async def test_list_all_returns_every_thread(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_threads(db_session)
    app = make_app([perm("threads", "read", "all")])
    async with http_client(app) as client:
        resp = await client.request("GET", _list_url(), json={})
    assert resp.status_code == 200, resp.text
    assert _ids(resp.json()) == {"own", "same-team", "other-team"}


@pytest.mark.asyncio
async def test_list_own_includes_is_shared_threads_in_team(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    """`_thread_scope_clause` ORs in is_shared rows for non-`.all` callers."""
    await _seed_thread(db_session, "own", user_id=SCOPE_USER_ID, team_id=SCOPE_TEAM_ID)
    await _seed_thread(
        db_session,
        "shared-by-teammate",
        user_id="teammate",
        team_id=SCOPE_TEAM_ID,
        is_shared=True,
    )
    await _seed_thread(
        db_session,
        "shared-other-team",
        user_id="stranger",
        team_id="other-team",
        is_shared=True,
    )
    app = make_app([perm("threads", "read", "own")])
    async with http_client(app) as client:
        resp = await client.request("GET", _list_url(), json={})
    assert resp.status_code == 200, resp.text
    # is_shared is gated by team in the second OR branch — cross-team must
    # not leak even when the row is flagged is_shared.
    assert _ids(resp.json()) == {"own", "shared-by-teammate"}


@pytest.mark.asyncio
async def test_get_other_users_thread_is_404_for_own_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_thread(
        db_session, "foreign", user_id="someone-else", team_id=SCOPE_TEAM_ID
    )
    app = make_app([perm("threads", "read", "own")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/threads/foreign"))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_other_users_thread_succeeds_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_thread(
        db_session, "foreign", user_id="someone-else", team_id=SCOPE_TEAM_ID
    )
    app = make_app([perm("threads", "read", "team")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/threads/foreign"))
    assert resp.status_code == 200, resp.text
    assert resp.json()["thread_id"] == "foreign"


@pytest.mark.asyncio
async def test_get_other_team_thread_is_404_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_thread(db_session, "foreign", user_id="x", team_id="other-team")
    app = make_app([perm("threads", "read", "team")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/threads/foreign"))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_other_team_thread_succeeds_for_all_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_thread(db_session, "foreign", user_id="x", team_id="other-team")
    app = make_app([perm("threads", "read", "all")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/threads/foreign"))
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_search_respects_team_scope(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_threads(db_session)
    app = make_app([perm("threads", "read", "team")])
    async with http_client(app) as client:
        resp = await client.post(tenant_url("/threads/search"), json={})
    assert resp.status_code == 200, resp.text
    ids = {t["thread_id"] for t in resp.json()}
    assert ids == {"own", "same-team"}


@pytest.mark.asyncio
async def test_delete_other_users_thread_is_404_for_own_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_thread(
        db_session, "foreign", user_id="someone-else", team_id=SCOPE_TEAM_ID
    )
    app = make_app([perm("threads", "delete", "own")])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/threads/foreign"))
    assert resp.status_code == 404


# --- Role-based tests ------------------------------------------------------
# Drives each endpoint with the full ROLE_PERMS list for a given role and
# checks the resulting access matches what that role should see.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "expected_ids"),
    [
        ("superadmin", {"own", "same-team", "other-team"}),
        ("admin", {"own", "same-team"}),
        ("user", {"own"}),
    ],
)
async def test_role_list_returns_expected_threads(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    expected_ids: set[str],
) -> None:
    await _seed_three_threads(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.request("GET", _list_url(), json={})
    assert resp.status_code == 200, resp.text
    assert _ids(resp.json()) == expected_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "thread_team", "expected_status"),
    [
        ("superadmin", "other-team", 200),
        ("superadmin", SCOPE_TEAM_ID, 200),
        ("admin", "other-team", 404),
        ("admin", SCOPE_TEAM_ID, 200),
        ("user", SCOPE_TEAM_ID, 404),
    ],
)
async def test_role_get_other_users_thread(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    thread_team: str,
    expected_status: int,
) -> None:
    await _seed_thread(
        db_session, "foreign", user_id="someone-else", team_id=thread_team
    )
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/threads/foreign"))
    assert resp.status_code == expected_status, resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "thread_team", "expected_status"),
    [
        ("superadmin", "other-team", 200),
        ("admin", "other-team", 404),
        ("admin", SCOPE_TEAM_ID, 200),
        ("user", SCOPE_TEAM_ID, 404),
    ],
)
async def test_role_delete_other_users_thread(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    thread_team: str,
    expected_status: int,
) -> None:
    await _seed_thread(
        db_session, "foreign", user_id="someone-else", team_id=thread_team
    )
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/threads/foreign"))
    assert resp.status_code == expected_status, resp.text
