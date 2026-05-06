"""Shared fixtures for permission integration tests."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path as FsPath
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import APIRouter, FastAPI, Path
from sqlalchemy import NullPool, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from aegra_api.core.auth_deps import get_current_user, require_auth
from aegra_api.core.orm import Tenant, get_current_tenant, get_session
from aegra_api.models.auth import User
from aegra_api.settings import settings


# --- Permission sets per legacy role ---------------------------------------

SUPERADMIN_PERMS: list[str] = [
    "threads.read.all",
    "threads.update.all",
    "threads.delete.all",
    "assistants.read.all",
    "assistants.update.all",
    "assistants.delete.all",
    "stores.read.all",
    "stores.update.all",
    "stores.delete.all",
    "runs.read.all",
    "runs.update.all",
    "runs.delete.all",
    "workflows.read.all",
    "workflows.update.all",
    "workflows.delete.all",
    "workflows.preview",
    "workflow_runs.read.all",
    "workflow_runs.update.all",
    "workflow_runs.delete.all",
    "workflow_schedules.read.all",
    "workflow_schedules.update.all",
    "workflow_schedules.delete.all",
    "control_plane.read.all",
    "control_plane.cleanup.all",
]

ADMIN_PERMS: list[str] = [
    "threads.read.team",
    "threads.update.team",
    "threads.delete.team",
    "assistants.read.team",
    "assistants.update.team",
    "assistants.delete.team",
    "stores.read.team",
    "stores.update.team",
    "stores.delete.team",
    "runs.read.team",
    "runs.update.team",
    "runs.delete.team",
    "workflows.preview",
    "workflows.read.team",
    "workflows.update.team",
    "workflows.delete.team",
    "workflow_runs.read.team",
    "workflow_runs.update.team",
    "workflow_runs.delete.team",
    "workflow_schedules.read.team",
    "workflow_schedules.update.team",
    "workflow_schedules.delete.team",
    "control_plane.read.team",
    "control_plane.cleanup.team",
]

USER_PERMS: list[str] = [
    "threads.read",
    "threads.update",
    "threads.delete",
    "assistants.read.team",
    "stores.read",
    "stores.update",
    "stores.delete",
    "runs.read",
    "runs.update",
    "runs.delete",
    "workflows.read",
    "workflows.update",
    "workflows.delete",
    "workflow_runs.read",
    "workflow_runs.update",
    "workflow_runs.delete",
    "workflow_schedules.read",
    "workflow_schedules.update",
    "workflow_schedules.delete",
    "workflows.preview",
]

ROLE_PERMS: dict[str, list[str]] = {
    "superadmin": SUPERADMIN_PERMS,
    "admin": ADMIN_PERMS,
    "user": USER_PERMS,
}


# --- Constants -------------------------------------------------------------

TENANT_UUID: str = "test-tenant"
SCOPE_USER_ID: str = "u-1"
SCOPE_TEAM_ID: str = "t-1"

TEST_SCHEMA: str = "prototype"
TEST_TENANT_UUID: str = "test-perms"


# --- Project auth.py loader -----------------------------------------------


def load_project_auth() -> Any:
    """Import the project-root ``auth.py`` and return its ``auth`` instance."""
    os.environ.setdefault("RAG_API_URL", "http://localhost:8000")
    # tests/fixtures/permissions.py → repo root is 4 parents up.
    repo_root = FsPath(__file__).resolve().parents[4]
    auth_path = repo_root / "auth.py"
    assert auth_path.is_file(), f"Project auth.py missing at {auth_path}"

    module_name = "_test_project_auth_module"
    spec = importlib.util.spec_from_file_location(module_name, str(auth_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.auth


@pytest.fixture(scope="session")
def project_auth() -> Any:
    return load_project_auth()


# --- Permission/user helpers ----------------------------------------------


def make_user(
    permissions: list[str],
    *,
    user_id: str = SCOPE_USER_ID,
    team_id: str = SCOPE_TEAM_ID,
) -> User:
    return User(id=user_id, team_id=team_id, permissions=permissions)


def tenant_url(path: str, tenant_uuid: str = TEST_TENANT_UUID) -> str:
    return f"/tenant/{tenant_uuid}{path}"


def expected_filters(reach: str) -> dict[str, str]:
    """Filter dict the auth handler is expected to return for a given reach."""
    if reach == "all":
        return {}
    if reach == "team":
        return {"team_id": SCOPE_TEAM_ID}
    return {"user_id": SCOPE_USER_ID, "team_id": SCOPE_TEAM_ID}


def perm(resource: str, action: str, reach: str) -> str:
    """Build a permission string. ``reach="own"`` omits the suffix."""
    base = f"{resource}.{action}"
    return base if reach == "own" else f"{base}.{reach}"


# --- DB connection / session fixtures -------------------------------------


@pytest_asyncio.fixture
async def db_conn() -> AsyncIterator[Any]:
    """Open a connection, start an outer transaction, roll back on teardown.

    The tenant row is inserted inside the outer transaction so the rollback
    cleans it up too.
    """
    engine = create_async_engine(settings.db.database_url, poolclass=NullPool)
    try:
        bound = engine.execution_options(schema_translate_map={None: TEST_SCHEMA})
        async with bound.connect() as conn:
            outer = await conn.begin()
            try:
                await conn.execute(
                    text(
                        "INSERT INTO public.tenants (uuid, schema, enabled) "
                        "VALUES (:u, :s, true) "
                        "ON CONFLICT (uuid) DO UPDATE SET enabled = true, schema = EXCLUDED.schema"
                    ),
                    {"u": TEST_TENANT_UUID, "s": TEST_SCHEMA},
                )
                yield conn
            finally:
                await outer.rollback()
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_conn: Any) -> AsyncIterator[AsyncSession]:
    """Seeding session bound to the per-test connection."""
    async with _new_session(db_conn) as session:
        yield session


def _new_session(db_conn: Any) -> AsyncSession:
    """Build an `AsyncSession` joined to the outer transaction via SAVEPOINTs."""
    return AsyncSession(
        bind=db_conn,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )


# --- App builder -----------------------------------------------------------


def make_db_app_builder(
    *,
    project_auth: Any,
    db_conn: Any,
    routers: Sequence[APIRouter],
    monkeypatch: pytest.MonkeyPatch,
    extra_overrides: Mapping[Any, Any] | None = None,
) -> Callable[..., FastAPI]:
    """Build the per-test factory that mounts `routers` under /tenant/{uuid}.

    Auth is stubbed (real `get_current_user` returns our test User), but the
    tenant lookup goes through the real `Tenant` row inserted by `db_conn`.
    The route's `get_session` is overridden to share `db_conn` so it sees
    seeded rows under the same outer transaction.
    """
    monkeypatch.setattr(
        "aegra_api.core.auth_handlers.get_auth_instance",
        lambda: project_auth,
    )

    async def _override_get_current_tenant(
        tenant_uuid: str = Path(...),
    ) -> Tenant | None:
        # Real lookup against the per-test connection — the production
        # `get_current_tenant` uses `db_manager`'s engine, which lives on a
        # different event loop and would cross-contaminate.
        async with _new_session(db_conn) as s:
            return await s.scalar(select(Tenant).where(Tenant.uuid == tenant_uuid))

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        async with _new_session(db_conn) as s:
            try:
                yield s
            except Exception:
                await s.rollback()
                raise

    def _build(
        permissions: list[str],
        *,
        user_id: str = SCOPE_USER_ID,
        team_id: str = SCOPE_TEAM_ID,
    ) -> FastAPI:
        app = FastAPI()
        tenant_router = APIRouter(prefix="/tenant/{tenant_uuid}")
        for r in routers:
            tenant_router.include_router(r)
        app.include_router(tenant_router)

        user = make_user(permissions, user_id=user_id, team_id=team_id)
        app.dependency_overrides[get_current_tenant] = _override_get_current_tenant
        app.dependency_overrides[require_auth] = lambda: user
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_session] = _override_get_session
        for dep, override in (extra_overrides or {}).items():
            app.dependency_overrides[dep] = override
        return app

    return _build


# --- HTTP client -----------------------------------------------------------


@contextlib.asynccontextmanager
async def http_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
