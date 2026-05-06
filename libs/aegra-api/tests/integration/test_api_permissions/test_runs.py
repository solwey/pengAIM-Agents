"""DB-backed permission tests for the runs router.

Runs scope by `(user_id, team_id)` like threads do, and `_run_scope_clause`
adds an OR for runs whose parent thread is `is_shared`. The test matrix
mirrors the threads tests: per-reach checks on each endpoint plus
role-parametrized cases driven by ``ROLE_PERMS``.

The streaming/executor-driven endpoints (`create_run`, `wait_for_run`,
`stream_run`, etc.) are out of scope here — they require a real
LangGraphService and Redis broker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from datetime import UTC, datetime
from unittest.mock import AsyncMock

from aegra_api.api import runs as runs_module
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models.runs import Run as RunModel
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
        routers=[runs_module.router],
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
            metadata_json={},
            user_id=user_id,
            team_id=team_id,
            is_shared=is_shared,
        )
    )
    await session.commit()


async def _seed_run(
    session: AsyncSession,
    run_id: str,
    *,
    thread_id: str,
    user_id: str,
    team_id: str,
    status: str = "success",
    assistant_id: str = "asst-test",
) -> None:
    session.add(
        RunORM(
            run_id=run_id,
            thread_id=thread_id,
            status=status,
            user_id=user_id,
            team_id=team_id,
            assistant_id=assistant_id,
        )
    )
    await session.commit()


async def _seed_assistant(session: AsyncSession) -> None:
    session.add(
        AssistantORM(
            assistant_id="asst-test",
            name="asst-test",
            graph_id="test-graph",
            team_id=SCOPE_TEAM_ID,
            config={},
            context={},
            metadata_dict={},
        )
    )
    await session.commit()


async def _seed_three_runs(session: AsyncSession) -> None:
    """One thread+run per scope: own / same-team / other-team.

    Thread ownership matches run ownership so the join in `_run_scope_clause`
    behaves naturally — the scope decision is driven by the run's columns.
    """
    await _seed_assistant(session)
    await _seed_thread(session, "t-own", user_id=SCOPE_USER_ID, team_id=SCOPE_TEAM_ID)
    await _seed_run(
        session, "r-own", thread_id="t-own", user_id=SCOPE_USER_ID, team_id=SCOPE_TEAM_ID
    )
    await _seed_thread(session, "t-team", user_id="other-user", team_id=SCOPE_TEAM_ID)
    await _seed_run(
        session, "r-team", thread_id="t-team", user_id="other-user", team_id=SCOPE_TEAM_ID
    )
    await _seed_thread(session, "t-other", user_id="other-user", team_id="other-team")
    await _seed_run(
        session,
        "r-other",
        thread_id="t-other",
        user_id="other-user",
        team_id="other-team",
    )


def _ids(payload: list[dict[str, Any]]) -> set[str]:
    return {r["run_id"] for r in payload}


def _history_ids(payload: dict[str, Any]) -> set[str]:
    return {r["run_id"] for r in payload["runs"]}


# --- Per-reach: list_runs (GET /threads/{tid}/runs) -----------------------


@pytest.mark.asyncio
async def test_list_runs_own_returns_only_callers_run(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "own")])
    async with http_client(app) as client:
        # Caller's own thread+run.
        resp_own = await client.get(tenant_url("/threads/t-own/runs"))
        # Same team, different user — must not appear under .own.
        resp_team = await client.get(tenant_url("/threads/t-team/runs"))
    assert resp_own.status_code == 200, resp_own.text
    assert _ids(resp_own.json()) == {"r-own"}
    assert resp_team.status_code == 200, resp_team.text
    assert _ids(resp_team.json()) == set()


@pytest.mark.asyncio
async def test_list_runs_team_returns_team_runs(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "team")])
    async with http_client(app) as client:
        resp_team = await client.get(tenant_url("/threads/t-team/runs"))
        resp_other = await client.get(tenant_url("/threads/t-other/runs"))
    assert _ids(resp_team.json()) == {"r-team"}
    assert _ids(resp_other.json()) == set()


@pytest.mark.asyncio
async def test_list_runs_all_returns_every_run(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "all")])
    async with http_client(app) as client:
        resp_other = await client.get(tenant_url("/threads/t-other/runs"))
    assert _ids(resp_other.json()) == {"r-other"}


@pytest.mark.asyncio
async def test_list_runs_own_includes_runs_on_shared_thread(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    """`_run_scope_clause` ORs in runs whose parent thread is `is_shared` (same team)."""
    await _seed_assistant(db_session)
    await _seed_thread(
        db_session,
        "t-shared",
        user_id="teammate",
        team_id=SCOPE_TEAM_ID,
        is_shared=True,
    )
    await _seed_run(
        db_session,
        "r-shared",
        thread_id="t-shared",
        user_id="teammate",
        team_id=SCOPE_TEAM_ID,
    )
    app = make_app([perm("runs", "read", "own")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/threads/t-shared/runs"))
    assert _ids(resp.json()) == {"r-shared"}


@pytest.mark.asyncio
async def test_list_runs_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("threads", "read", "all")])  # unrelated perm
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/threads/t-own/runs"))
    assert resp.status_code == 403


# --- Per-reach: get_run (GET /threads/{tid}/runs/{rid}) -------------------


@pytest.mark.asyncio
async def test_get_run_other_user_is_404_for_own_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "own")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/threads/t-team/runs/r-team"))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_run_other_user_succeeds_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "team")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/threads/t-team/runs/r-team"))
    assert resp.status_code == 200, resp.text
    assert resp.json()["run_id"] == "r-team"


@pytest.mark.asyncio
async def test_get_run_other_team_is_404_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "team")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/threads/t-other/runs/r-other"))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_run_other_team_succeeds_for_all_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "all")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/threads/t-other/runs/r-other"))
    assert resp.status_code == 200, resp.text


# --- Per-reach: delete_run (DELETE /threads/{tid}/runs/{rid}) -------------


@pytest.mark.asyncio
async def test_delete_run_other_user_is_404_for_own_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "delete", "own")])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/threads/t-team/runs/r-team"))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_run_team_succeeds_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "delete", "team")])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/threads/t-team/runs/r-team"))
    assert resp.status_code == 204, resp.text


@pytest.mark.asyncio
async def test_delete_run_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "all")])  # no delete perm
    async with http_client(app) as client:
        resp = await client.delete(tenant_url("/threads/t-own/runs/r-own"))
    assert resp.status_code == 403


# --- Per-reach: list_runs_history (GET /runs) -----------------------------


@pytest.mark.asyncio
async def test_list_runs_history_team_returns_team_runs(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    """`/runs` history uses `get_scope_filters` directly (no `is_shared` carve-out)."""
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "team")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/runs"))
    assert resp.status_code == 200, resp.text
    assert _history_ids(resp.json()) == {"r-own", "r-team"}


@pytest.mark.asyncio
async def test_list_runs_history_all_returns_every_run(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "all")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/runs"))
    assert _history_ids(resp.json()) == {"r-own", "r-team", "r-other"}


@pytest.mark.asyncio
async def test_list_runs_history_own_returns_only_callers_runs(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "own")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/runs"))
    assert _history_ids(resp.json()) == {"r-own"}


# --- Per-reach: get_run_detail / get_run_status_history / get_run_events --


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    ["/runs/r-other/detail", "/runs/r-other/history"],
)
async def test_run_detail_endpoints_404_for_team_reach_on_other_team(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    endpoint: str,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "team")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url(endpoint))
    assert resp.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    ["/runs/r-other/detail", "/runs/r-other/history"],
)
async def test_run_detail_endpoints_succeed_for_all_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    endpoint: str,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "all")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url(endpoint))
    assert resp.status_code == 200, resp.text


# --- Role-based tests ------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "expected_ids"),
    [
        ("superadmin", {"r-own", "r-team", "r-other"}),
        ("admin", {"r-own", "r-team"}),
        ("user", {"r-own"}),
    ],
)
async def test_role_list_runs_history_returns_expected_runs(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    expected_ids: set[str],
) -> None:
    await _seed_three_runs(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/runs"))
    assert resp.status_code == 200, resp.text
    assert _history_ids(resp.json()) == expected_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "thread_id", "run_id", "expected_status"),
    [
        ("superadmin", "t-other", "r-other", 200),
        ("superadmin", "t-team", "r-team", 200),
        ("admin", "t-other", "r-other", 404),
        ("admin", "t-team", "r-team", 200),
        ("user", "t-team", "r-team", 404),
        ("user", "t-own", "r-own", 200),
    ],
)
async def test_role_get_run(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    thread_id: str,
    run_id: str,
    expected_status: int,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.get(tenant_url(f"/threads/{thread_id}/runs/{run_id}"))
    assert resp.status_code == expected_status, resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "thread_id", "run_id", "expected_status"),
    [
        ("superadmin", "t-other", "r-other", 204),
        ("admin", "t-other", "r-other", 404),
        ("admin", "t-team", "r-team", 204),
        ("user", "t-team", "r-team", 404),
        ("user", "t-own", "r-own", 204),
    ],
)
async def test_role_delete_run(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    role: str,
    thread_id: str,
    run_id: str,
    expected_status: int,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.delete(tenant_url(f"/threads/{thread_id}/runs/{run_id}"))
    assert resp.status_code == expected_status, resp.text


# --- Streaming / executor endpoints (mocked) ------------------------------
# These routes go through `_prepare_run` (which talks to LangGraphService and
# the executor) or `streaming_service` (which wraps the SSE broker). For
# permission tests we don't care about real graph execution — we mock those
# integrations so the auth gate is the only thing that varies.


def _fake_run_model(thread_id: str, run_id: str = "fake-run") -> RunModel:
    now = datetime.now(UTC)
    return RunModel(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id="asst-test",
        status="pending",
        input={},
        user_id=SCOPE_USER_ID,
        team_id=SCOPE_TEAM_ID,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def mock_executor(monkeypatch: pytest.MonkeyPatch, db_conn: Any) -> None:
    """Stub out the run executor and streaming broker.

    `_prepare_run` and the streaming-service methods are imported by name in
    `runs_module`, so we patch them on that module. `get_session` is also
    re-imported by name and called manually inside `wait_for_run`; we swap
    it for a generator that yields a real session bound to the per-test
    `db_conn` so the post-auth thread-scope check can read seeded rows.
    """
    from tests.fixtures.permissions import _new_session

    async def _fake_prepare(
        session: Any,
        thread_id: str,
        request: Any,
        user: Any,
        tenant: Any,
        *,
        initial_status: str,
    ) -> tuple[str, RunModel, object]:
        return ("fake-run", _fake_run_model(thread_id=thread_id), object())

    async def _fake_get_session(tenant: Any = None) -> Any:
        async with _new_session(db_conn) as s:
            yield s

    async def _fake_stream(*args: Any, **kwargs: Any) -> Any:
        if False:
            yield ""

    monkeypatch.setattr(runs_module, "_prepare_run", _fake_prepare)
    monkeypatch.setattr(runs_module, "get_session", _fake_get_session)
    monkeypatch.setattr(runs_module.streaming_service, "interrupt_run", AsyncMock())
    monkeypatch.setattr(runs_module.streaming_service, "cancel_run", AsyncMock())
    monkeypatch.setattr(
        runs_module.streaming_service, "stream_run_execution", _fake_stream
    )


def _run_create_body() -> dict[str, Any]:
    return {"assistant_id": "asst-test", "input": {}}


# --- create_run (POST /threads/{tid}/runs) --------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reach", "thread_id", "expected_status"),
    [
        # In-scope threads succeed.
        ("own", "t-own", 200),
        ("team", "t-team", 200),
        ("all", "t-other", 200),
        # Out-of-scope threads return 404 via `_ensure_thread_in_scope`.
        ("own", "t-team", 404),
        ("team", "t-other", 404),
    ],
)
async def test_create_run_scoped_by_thread_visibility(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
    reach: str,
    thread_id: str,
    expected_status: int,
) -> None:
    await _seed_three_runs(db_session)
    # `create_run` is gated by threads.create_run → maps to runs.update in auth.
    app = make_app([perm("runs", "update", reach)])
    async with http_client(app) as client:
        resp = await client.post(
            tenant_url(f"/threads/{thread_id}/runs"), json=_run_create_body()
        )
    assert resp.status_code == expected_status, resp.text


@pytest.mark.asyncio
async def test_create_run_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "all")])  # no update perm
    async with http_client(app) as client:
        resp = await client.post(
            tenant_url("/threads/t-own/runs"), json=_run_create_body()
        )
    assert resp.status_code == 403


# --- create_and_stream_run (POST /threads/{tid}/runs/stream) --------------


@pytest.mark.asyncio
async def test_create_and_stream_run_team_succeeds(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "update", "team")])
    async with http_client(app) as client:
        resp = await client.post(
            tenant_url("/threads/t-team/runs/stream"), json=_run_create_body()
        )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")


@pytest.mark.asyncio
async def test_create_and_stream_run_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "all")])
    async with http_client(app) as client:
        resp = await client.post(
            tenant_url("/threads/t-own/runs/stream"), json=_run_create_body()
        )
    assert resp.status_code == 403


# --- wait_for_run (POST /threads/{tid}/runs/wait) -------------------------


@pytest.mark.asyncio
async def test_wait_for_run_team_succeeds(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "update", "team")])
    async with http_client(app) as client:
        resp = await client.post(
            tenant_url("/threads/t-team/runs/wait"), json=_run_create_body()
        )
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_wait_for_run_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "all")])
    async with http_client(app) as client:
        resp = await client.post(
            tenant_url("/threads/t-own/runs/wait"), json=_run_create_body()
        )
    assert resp.status_code == 403


# --- stream_run (GET /threads/{tid}/runs/{rid}/stream) --------------------
# Seeded with status="success" → terminal path emits a single end event
# without invoking `streaming_service`.


@pytest.mark.asyncio
async def test_stream_run_terminal_team_succeeds(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "team")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/threads/t-team/runs/r-team/stream"))
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")


@pytest.mark.asyncio
async def test_stream_run_other_team_is_404_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "team")])
    async with http_client(app) as client:
        resp = await client.get(tenant_url("/threads/t-other/runs/r-other/stream"))
    assert resp.status_code == 404


# --- update_run (PATCH /threads/{tid}/runs/{rid}) -------------------------
# Body status is non-interrupted → just refetches the run, no streaming.


@pytest.mark.asyncio
async def test_update_run_other_user_is_404_for_own_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "update", "own")])
    async with http_client(app) as client:
        resp = await client.patch(
            tenant_url("/threads/t-team/runs/r-team"),
            json={"run_id": "r-team", "status": "success"},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_run_team_succeeds_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "update", "team")])
    async with http_client(app) as client:
        resp = await client.patch(
            tenant_url("/threads/t-team/runs/r-team"),
            json={"run_id": "r-team", "status": "success"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["run_id"] == "r-team"


@pytest.mark.asyncio
async def test_update_run_missing_permission_is_403(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "read", "all")])
    async with http_client(app) as client:
        resp = await client.patch(
            tenant_url("/threads/t-own/runs/r-own"),
            json={"run_id": "r-own", "status": "success"},
        )
    assert resp.status_code == 403


# --- cancel_run_endpoint (POST /threads/{tid}/runs/{rid}/cancel) ----------


@pytest.mark.asyncio
async def test_cancel_run_endpoint_team_succeeds(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "update", "team")])
    async with http_client(app) as client:
        resp = await client.post(tenant_url("/threads/t-team/runs/r-team/cancel"))
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_cancel_run_endpoint_other_team_is_404_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "update", "team")])
    async with http_client(app) as client:
        resp = await client.post(tenant_url("/threads/t-other/runs/r-other/cancel"))
    assert resp.status_code == 404


# --- cancel_run_by_id (POST /runs/{rid}/cancel) ---------------------------
# Status must be pending/running for the cancel to apply (else 409). Seed a
# fresh "running" run so this exercises the success path.


@pytest.mark.asyncio
async def test_cancel_run_by_id_team_succeeds(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_assistant(db_session)
    await _seed_thread(
        db_session, "t-running", user_id="other-user", team_id=SCOPE_TEAM_ID
    )
    await _seed_run(
        db_session,
        "r-running",
        thread_id="t-running",
        user_id="other-user",
        team_id=SCOPE_TEAM_ID,
        status="running",
    )
    app = make_app([perm("runs", "update", "team")])
    async with http_client(app) as client:
        resp = await client.post(tenant_url("/runs/r-running/cancel"))
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "interrupted", "run_id": "r-running"}


@pytest.mark.asyncio
async def test_cancel_run_by_id_other_team_is_404_for_team_reach(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app([perm("runs", "update", "team")])
    async with http_client(app) as client:
        resp = await client.post(tenant_url("/runs/r-other/cancel"))
    assert resp.status_code == 404


# --- Role-based tests for streaming/executor endpoints --------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "thread_id", "expected_status"),
    [
        ("superadmin", "t-other", 200),
        ("superadmin", "t-team", 200),
        ("admin", "t-other", 404),
        ("admin", "t-team", 200),
        ("user", "t-team", 404),
        ("user", "t-own", 200),
    ],
)
async def test_role_create_run(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
    role: str,
    thread_id: str,
    expected_status: int,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.post(
            tenant_url(f"/threads/{thread_id}/runs"), json=_run_create_body()
        )
    assert resp.status_code == expected_status, resp.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "thread_id", "run_id", "expected_status"),
    [
        ("superadmin", "t-other", "r-other", 200),
        ("admin", "t-other", "r-other", 404),
        ("admin", "t-team", "r-team", 200),
        ("user", "t-team", "r-team", 404),
        ("user", "t-own", "r-own", 200),
    ],
)
async def test_role_cancel_run_endpoint(
    make_app: Callable[..., FastAPI],
    db_session: AsyncSession,
    mock_executor: None,
    role: str,
    thread_id: str,
    run_id: str,
    expected_status: int,
) -> None:
    await _seed_three_runs(db_session)
    app = make_app(ROLE_PERMS[role])
    async with http_client(app) as client:
        resp = await client.post(
            tenant_url(f"/threads/{thread_id}/runs/{run_id}/cancel")
        )
    assert resp.status_code == expected_status, resp.text
