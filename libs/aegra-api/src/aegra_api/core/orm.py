"""SQLAlchemy ORM setup for persistent assistant/thread/run records.

This module creates:
• `Base` – the declarative base used by our models.
• `Assistant`, `Thread`, `Run` – ORM models mirroring the bootstrap tables
  already created in ``DatabaseManager._create_metadata_tables``.
• `async_session_maker` – a factory that hands out `AsyncSession` objects
  bound to the shared engine managed by `db_manager`.
• `get_session` – FastAPI dependency helper for routers.

Nothing is auto-imported by FastAPI yet; routers will `from ...core.db import get_session`.
"""

from collections.abc import AsyncIterator
from contextvars import ContextVar
from datetime import datetime

from fastapi import Depends, Path
from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

Base = declarative_base()


# Context variable holding the currently-active tenant for the request.
# Set by `get_current_tenant` dependency; read by logging / background tasks.
tenant_var: ContextVar["Tenant | None"] = ContextVar("tenant", default=None)


class Tenant(Base):
    """Tenant registry. Lives in the public schema."""

    __tablename__ = "tenants"
    __table_args__ = (
        UniqueConstraint("schema", name="uq_tenant_schema"),
        {"schema": "public"},
    )

    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)
    schema: Mapped[str] = mapped_column(String(63), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=text("now()"))


class Assistant(Base):
    __tablename__ = "assistant"

    # TEXT PK with DB-side generation using uuid_generate_v4()::text
    assistant_id: Mapped[str] = mapped_column(
        Text, primary_key=True, server_default=text("public.uuid_generate_v4()::text")
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    graph_id: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    context: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    team_id: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    metadata_dict: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"), name="metadata")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    type: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Indexes for performance
    __table_args__ = (
        Index("idx_assistant_user", "team_id"),
        Index("idx_assistant_user_assistant", "team_id", "assistant_id", unique=True),
        Index("idx_assistant_deleted_at", "deleted_at"),
        Index("idx_assistant_enabled", "enabled"),
        Index("idx_assistant_type", "type"),
    )


class AssistantVersion(Base):
    __tablename__ = "assistant_versions"

    assistant_id: Mapped[str] = mapped_column(
        Text, ForeignKey("assistant.assistant_id", ondelete="CASCADE"), primary_key=True
    )
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    graph_id: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict | None] = mapped_column(JSONB)
    context: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    metadata_dict: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"), name="metadata")
    name: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)


class Thread(Base):
    __tablename__ = "thread"

    thread_id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text, server_default=text("'idle'"))
    # Database column is 'metadata_json' (per database.py). ORM attribute 'metadata_json' must map to that column.
    metadata_json: Mapped[dict] = mapped_column("metadata_json", JSONB, server_default=text("'{}'::jsonb"))
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    team_id: Mapped[str] = mapped_column(Text, nullable=False)
    is_shared: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default=text("false"))
    assistant_id: Mapped[str | None] = mapped_column(Text, ForeignKey("assistant.assistant_id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    # Indexes for performance
    __table_args__ = (
        Index("idx_thread_user", "user_id"),
        Index("idx_thread_team", "team_id"),
        Index("idx_thread_assistant", "assistant_id"),
        Index("idx_thread_team_user", "team_id", "user_id"),
        Index("idx_thread_team_assistant", "team_id", "assistant_id"),
        Index("idx_thread_user_assistant", "user_id", "assistant_id"),
        Index("idx_thread_team_is_shared", "team_id", "is_shared"),
        Index(
            "idx_thread_team_assistant_created_at",
            "team_id",
            "assistant_id",
            "created_at",
        ),
        Index("idx_thread_deleted_at", "deleted_at"),
    )


class Run(Base):
    __tablename__ = "runs"

    # TEXT PK with DB-side generation using uuid_generate_v4()::text
    run_id: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("public.uuid_generate_v4()::text"))
    thread_id: Mapped[str] = mapped_column(Text, ForeignKey("thread.thread_id", ondelete="CASCADE"), nullable=False)
    assistant_id: Mapped[str | None] = mapped_column(Text, ForeignKey("assistant.assistant_id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(Text, server_default=text("'pending'"))
    input: Mapped[dict | None] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    # Some environments may not yet have a 'config' column; make it nullable without default to match existing DB.
    # If migrations add this column later, it's already represented here.
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    output: Mapped[dict | None] = mapped_column(JSONB)
    error_message: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    current_step: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    team_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    tool_calls_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tools_used: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    config_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Worker execution: stores RunJob params so workers can reconstruct
    # the job from the database after receiving a run_id via Redis.
    execution_params: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Lease-based crash recovery: tracks which worker owns a run and
    # when the lease expires. A background reaper re-enqueues runs
    # whose leases have expired (worker crashed).
    claimed_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    # Indexes for performance
    __table_args__ = (
        Index("idx_runs_thread_id", "thread_id"),
        Index("idx_runs_user", "user_id"),
        Index("idx_runs_status", "status"),
        Index("idx_runs_assistant_id", "assistant_id"),
        Index("idx_runs_created_at", "created_at"),
        Index("idx_runs_lease_reaper", "status", "lease_expires_at"),
        Index("idx_runs_team", "team_id"),
        Index("idx_runs_team_user", "team_id", "user_id"),
        Index("idx_runs_thread_team_created_at", "thread_id", "team_id", "created_at"),
        Index(
            "idx_runs_team_assistant_created_at",
            "team_id",
            "assistant_id",
            "created_at",
        ),
    )


class WorkerHeartbeat(Base):
    __tablename__ = "worker_heartbeat"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    status: Mapped[str] = mapped_column(Text, server_default=text("'online'"))
    started_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    last_heartbeat: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    active_run_count: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    metadata_dict: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"), name="metadata")


class RunStatusHistory(Base):
    __tablename__ = "run_status_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(Text, ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False)
    from_status: Mapped[str | None] = mapped_column(Text, nullable=True)
    to_status: Mapped[str] = mapped_column(Text, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    traceback: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))

    __table_args__ = (Index("idx_run_status_history_run_id_created_at", "run_id", "created_at"),)


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))

    # Indexes for performance
    __table_args__ = (
        Index("idx_run_events_run_id", "run_id"),
        Index("idx_run_events_seq", "run_id", "seq"),
    )


class Workflow(Base):
    __tablename__ = "workflows"

    id: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("uuid_generate_v4()::text"))
    team_id: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    definition: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    webhook_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    webhook_path: Mapped[str | None] = mapped_column(Text, nullable=True, unique=True)
    webhook_secret: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("idx_workflow_team", "team_id"),
        Index("idx_workflow_team_user", "team_id", "user_id"),
        Index("idx_workflow_is_active", "is_active"),
        Index("idx_workflow_deleted_at", "deleted_at"),
        Index("idx_workflow_webhook_path", "webhook_path", unique=True),
    )


class WorkflowVersion(Base):
    __tablename__ = "workflow_versions"

    workflow_id: Mapped[str] = mapped_column(Text, ForeignKey("workflows.id", ondelete="CASCADE"), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    definition: Mapped[dict] = mapped_column(JSONB, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("uuid_generate_v4()::text"))
    workflow_id: Mapped[str] = mapped_column(Text, ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False)
    team_id: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'pending'"))
    input_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    output_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_workflow_run_workflow", "workflow_id"),
        Index("idx_workflow_run_team", "team_id"),
        Index("idx_workflow_run_status", "status"),
        Index("idx_workflow_run_team_workflow", "team_id", "workflow_id"),
        Index("idx_workflow_run_created_at", "created_at"),
    )


class WorkflowSchedule(Base):
    __tablename__ = "workflow_schedules"

    id: Mapped[str] = mapped_column(Text, primary_key=True, server_default=text("uuid_generate_v4()::text"))
    workflow_id: Mapped[str] = mapped_column(Text, ForeignKey("workflows.id", ondelete="CASCADE"), nullable=False)
    team_id: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    cron_expression: Mapped[str] = mapped_column(Text, nullable=False)
    timezone: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'UTC'"))
    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    input_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    last_run_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    next_run_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=text("now()"))

    __table_args__ = (
        Index("idx_schedule_workflow", "workflow_id"),
        Index("idx_schedule_team", "team_id"),
        Index("idx_schedule_enabled_next", "is_enabled", "next_run_at"),
        Index(
            "idx_schedule_unique_active_workflow",
            "workflow_id",
            unique=True,
            postgresql_where=text("is_enabled = true"),
        ),
    )


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

async_session_maker: async_sessionmaker[AsyncSession] | None = None


def _get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Return a cached async_sessionmaker bound to db_manager.engine."""
    global async_session_maker
    if async_session_maker is None:
        from aegra_api.core.database import db_manager

        engine = db_manager.get_engine()
        async_session_maker = async_sessionmaker(engine, expire_on_commit=False)
    return async_session_maker


async def get_session_public() -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession bound to the public schema."""
    maker = _get_session_maker()
    async with maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def get_current_tenant(
    tenant_uuid: str = Path(..., description="Tenant UUID"),
) -> "Tenant | None":
    """Resolve the current request's tenant from the path parameter."""
    cached = tenant_var.get()
    if cached is not None and cached.uuid == tenant_uuid:
        return cached

    maker = _get_session_maker()
    async with maker() as db:
        result = await db.execute(select(Tenant).where(Tenant.uuid == tenant_uuid))
        tenant = result.scalar_one_or_none()

    tenant_var.set(tenant)
    return tenant


def new_tenant_session(schema: str) -> AsyncSession:
    """Return a fresh ``AsyncSession`` bound to a tenant schema."""
    from aegra_api.core.database import db_manager

    if not schema:
        raise RuntimeError("new_tenant_session requires a non-empty schema")

    engine = db_manager.get_engine().execution_options(schema_translate_map={None: schema})
    return AsyncSession(bind=engine, expire_on_commit=False)


async def get_session(
    tenant: "Tenant" = Depends(get_current_tenant),
) -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession bound to the current tenant's schema."""
    if tenant is None:
        # `validate_tenant` should have already rejected this, but guard
        # in case a route uses `get_session` without `validate_tenant`.
        raise RuntimeError("get_session called without a valid tenant")

    session = new_tenant_session(tenant.schema)
    async with session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
