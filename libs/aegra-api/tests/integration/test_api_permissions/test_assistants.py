"""DB-backed permission tests for the assistants router.

Assistants scope by `team_id` only (no per-user column) and always include
the special ``team_id == "system"`` row so built-in assistants stay
visible to every team. The service layer (`_scope_assistants`) builds the
WHERE clause from the auth filter dict; these tests verify the rows that
actually come back.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from aegra_api.api import assistants as assistants_module
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.services.assistant_service import (
    AssistantService,
    get_assistant_service,
)
from tests.fixtures.permissions import (
    ROLE_PERMS,
    SCOPE_TEAM_ID,
    http_client,
    make_db_app_builder,
    perm,
    tenant_url,
)


@pytest.fixture
def make_app(
    project_auth: Any,
    db_conn: Any,
    db_session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., FastAPI]:
    # `get_assistant_service` would normally construct a real LangGraphService
    # (loads graphs from aegra.json). For these tests we only exercise the DB
    # query paths, so a mock is sufficient.
    def _override_assistant_service() -> AssistantService:
        return AssistantService(session=db_session, langgraph_service=AsyncMock())

    return make_db_app_builder(
        project_auth=project_auth,
        db_conn=db_conn,
        routers=[assistants_module.router],
        monkeypatch=monkeypatch,
        extra_overrides={get_assistant_service: _override_assistant_service},
    )


# --- Seed helpers ----------------------------------------------------------


async def _seed_assistant(
    session: AsyncSession,
    assistant_id: str,
    *,
    team_id: str,
    name: str | None = None,
) -> None:
    session.add(
        AssistantORM(
            assistant_id=assistant_id,
            name=name or assistant_id,
            graph_id="test-graph",
            team_id=team_id,
            config={},
            context={},
            metadata_dict={},
        )
    )
    await session.commit()


async def _seed_three_assistants(session: AsyncSession) -> None:
    await _seed_assistant(session, "own-team", team_id=SCOPE_TEAM_ID)
    await _seed_assistant(session, "system", team_id="system")
    await _seed_assistant(session, "other-team", team_id="other-team")


def _list_url() -> str:
    return tenant_url("/assistants")


def _ids_from_list(payload: dict[str, Any]) -> set[str]:
    return {a["assistant_id"] for a in payload["assistants"]}


def _ids_from_search(payload: list[dict[str, Any]]) -> set[str]:
    return {a["assistant_id"] for a in payload}


# --- Tests -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_team_returns_team_plus_system(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_assistants(db_session)
    app = make_app([perm("assistants", "read", "team")])
    async with http_client(app) as client:
        resp = await client.get(_list_url())
    assert resp.status_code == 200, resp.text
    assert _ids_from_list(resp.json()) == {"own-team", "system"}


@pytest.mark.asyncio
async def test_list_own_returns_team_plus_system(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    """Assistants have no user_id column — `own` and `team` reach behave the same."""
    await _seed_three_assistants(db_session)
    app = make_app([perm("assistants", "read", "own")])
    async with http_client(app) as client:
        resp = await client.get(_list_url())
    assert resp.status_code == 200, resp.text
    assert _ids_from_list(resp.json()) == {"own-team", "system"}


@pytest.mark.asyncio
async def test_list_all_returns_every_assistant(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_assistants(db_session)
    app = make_app([perm("assistants", "read", "all")])
    async with http_client(app) as client:
        resp = await client.get(_list_url())
    assert resp.status_code == 200, resp.text
    assert _ids_from_list(resp.json()) == {"own-team", "system", "other-team"}


@pytest.mark.asyncio
async def test_search_team_returns_team_plus_system(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_assistants(db_session)
    app = make_app([perm("assistants", "read", "team")])
    async with http_client(app) as client:
        resp = await client.post(tenant_url("/assistants/search"), json={})
    assert resp.status_code == 200, resp.text
    assert _ids_from_search(resp.json()) == {"own-team", "system"}


@pytest.mark.asyncio
async def test_get_other_team_assistant_is_404_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_assistant(db_session, "other", team_id="other-team")
    app = make_app([perm("assistants", "read", "team")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/assistants/other"))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_other_team_assistant_succeeds_for_all_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_assistant(db_session, "other", team_id="other-team")
    app = make_app([perm("assistants", "read", "all")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/assistants/other"))
    assert resp.status_code == 200, resp.text
    assert resp.json()["assistant_id"] == "other"


@pytest.mark.asyncio
async def test_get_system_assistant_succeeds_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    """The `team_id == "system"` carve-out makes built-ins visible to every team."""
    await _seed_assistant(db_session, "system-asst", team_id="system")
    app = make_app([perm("assistants", "read", "team")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/assistants/system-asst"))
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_count_team_counts_team_plus_system(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_assistants(db_session)
    app = make_app([perm("assistants", "read", "team")])
    async with http_client(app) as client:
        resp = await client.post(tenant_url("/assistants/count"), json={})
    assert resp.status_code == 200, resp.text
    assert resp.json() == 2


@pytest.mark.asyncio
async def test_delete_other_team_assistant_is_404_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_assistant(db_session, "other", team_id="other-team")
    app = make_app([perm("assistants", "delete", "team")])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/assistants/other"))
    assert resp.status_code == 404


# --- Role-based tests ------------------------------------------------------
# Drives each endpoint with the full ROLE_PERMS list for a given role and
# checks the resulting access matches what that role should see.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "expected_ids"),
    [
        ("superadmin", {"own-team", "system", "other-team"}),
        ("admin", {"own-team", "system"}),
        ("user", {"own-team", "system"}),
    ],
)
async def test_role_list_returns_expected_assistants(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    expected_ids: set[str],
) -> None:
    await _seed_three_assistants(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.get(_list_url())
    assert resp.status_code == 200, resp.text
    assert _ids_from_list(resp.json()) == expected_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "asst_team", "expected_status"),
    [
        ("superadmin", "other-team", 200),
        ("superadmin", SCOPE_TEAM_ID, 200),
        ("admin", "other-team", 404),
        ("admin", SCOPE_TEAM_ID, 200),
        ("user", "other-team", 404),
        ("user", SCOPE_TEAM_ID, 200),
        ("user", "system", 200),
    ],
)
async def test_role_get_assistant(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    asst_team: str,
    expected_status: int,
) -> None:
    await _seed_assistant(db_session, "target", team_id=asst_team)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/assistants/target"))
    assert resp.status_code == expected_status, resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "asst_team", "expected_status"),
    [
        ("superadmin", "other-team", 200),
        ("admin", "other-team", 404),
        ("admin", SCOPE_TEAM_ID, 200),
        # `user` role has no `assistants.delete*` permission at all → denied.
        ("user", SCOPE_TEAM_ID, 403),
    ],
)
async def test_role_delete_assistant(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    asst_team: str,
    expected_status: int,
) -> None:
    await _seed_assistant(db_session, "target", team_id=asst_team)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/assistants/target"))
    assert resp.status_code == expected_status, resp.text
