"""Database manager with LangGraph integration"""

import asyncio

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from aegra_api.config import load_store_config
from aegra_api.settings import settings

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """Manages database connections and LangGraph persistence components"""

    def __init__(self) -> None:
        self.engine: AsyncEngine | None = None
        self.sync_engine: Engine | None = None

        # Shared pool for LangGraph components (Checkpointer + Store)
        self.lg_pool: AsyncConnectionPool | None = None
        self._checkpointer: AsyncPostgresSaver | None = None
        self._store: AsyncPostgresStore | None = None
        self._database_url = settings.db.database_url
        self._langgraph_lock = asyncio.Lock()

    def _build_sync_engine(self) -> Engine:
        """Build sync SQLAlchemy engine with psycopg3 driver."""
        sync_sqlalchemy_url = settings.db.database_url_sync.replace(
            "postgresql://",
            "postgresql+psycopg://",
            1,
        )
        return create_engine(
            sync_sqlalchemy_url,
            pool_pre_ping=True,
            echo=settings.db.DB_ECHO_LOG,
            connect_args={
                "prepare_threshold": None
            },  # psycopg3: disable server-side prepared statements (PgBouncer compat)
        )

    def ensure_async_engine(self) -> None:
        """Initialize async SQLAlchemy engine lazily."""
        if self.engine is None:
            # We strictly limit this pool because the main load
            # is handled by LangGraph components.
            self.engine = create_async_engine(
                self._database_url,
                pool_size=settings.pool.SQLALCHEMY_POOL_SIZE,
                max_overflow=settings.pool.SQLALCHEMY_MAX_OVERFLOW,
                pool_pre_ping=True,
                echo=settings.db.DB_ECHO_LOG,
            )

    def ensure_sync_engine(self) -> None:
        """Initialize only sync SQLAlchemy engine (no async/langgraph side effects)."""
        if self.sync_engine is None:
            self.sync_engine = self._build_sync_engine()

    async def initialize(self) -> None:
        """Initialize database connections and LangGraph components"""
        # 1. Async SQLAlchemy engine (lazy/idempotent)
        self.ensure_async_engine()

        # 2. Ensure LangGraph components are initialized
        await self._ensure_langgraph_components()

        # Keep startup logs unchanged/compact
        if self._store is not None:
            store_config = load_store_config()
            index_config = store_config.get("index") if store_config else None
            if index_config:
                embed_model = index_config.get("embed", "unknown")
                logger.info(f"Semantic store enabled with embeddings: {embed_model}")

        logger.info("✅ Database and LangGraph components initialized")

    async def _create_langgraph_components(self) -> None:
        """Create shared pool + LangGraph checkpointer/store."""
        lg_max = settings.pool.LANGGRAPH_MAX_POOL_SIZE
        lg_kwargs = {
            "autocommit": True,
            "prepare_threshold": None,  # Disable prepared statements for PgBouncer compatibility
            "row_factory": dict_row,  # LangGraph requires dictionary rows, not tuples
        }

        logger.info(f"Initializing LangGraph components with shared pool (max {lg_max} conns)...")

        self.lg_pool = AsyncConnectionPool(
            conninfo=settings.db.database_url_sync,
            min_size=settings.pool.LANGGRAPH_MIN_POOL_SIZE,
            max_size=lg_max,
            open=False,
            kwargs=lg_kwargs,
            check=AsyncConnectionPool.check_connection,
        )

        # Explicitly open the pool
        await self.lg_pool.open()

        self._checkpointer = AsyncPostgresSaver(conn=self.lg_pool)
        await self._checkpointer.setup()  # Ensure tables exist

        # Load store configuration for semantic search (if configured)
        store_config = load_store_config()
        index_config = store_config.get("index") if store_config else None

        self._store = AsyncPostgresStore(conn=self.lg_pool, index=index_config)
        await self._store.setup()  # Ensure tables exist

    async def _reset_langgraph_components(self) -> None:
        """Best-effort reset of LangGraph resources."""
        if self.lg_pool is not None:
            try:
                await self.lg_pool.close()
            except Exception as e:
                logger.warning("Failed to close LangGraph pool during reset", error=str(e))

        self.lg_pool = None
        self._checkpointer = None
        self._store = None

    def _langgraph_healthy(self) -> bool:
        """Quick health signal for already-initialized components."""
        if self.lg_pool is None or self._checkpointer is None or self._store is None:
            return False
        return not self.lg_pool.closed

    async def _ensure_langgraph_components(self) -> None:
        """Ensure LangGraph components are available; auto-recover on closure/errors."""
        if self._langgraph_healthy():
            return

        async with self._langgraph_lock:
            if self._langgraph_healthy():
                return

            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    await self._reset_langgraph_components()
                    await self._create_langgraph_components()
                    if attempt > 0:
                        logger.info("LangGraph components recovered after retry")
                    return
                except Exception as e:
                    last_error = e
                    msg = str(e).lower()
                    is_connection_error = any(token in msg for token in ("closed", "connection", "timeout", "pool"))
                    logger.warning(
                        "Failed to initialize LangGraph components",
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    if attempt == 0 and is_connection_error:
                        continue
                    break

            raise RuntimeError("Failed to initialize/recover LangGraph components") from last_error

    async def close(self) -> None:
        """Close database connections"""
        # Close SQLAlchemy engine
        if self.engine:
            await self.engine.dispose()
            self.engine = None
        if self.sync_engine:
            self.sync_engine.dispose()
            self.sync_engine = None

        # Close shared LangGraph pool
        await self._reset_langgraph_components()

        logger.info("✅ Database connections closed")

    async def get_checkpointer(self) -> AsyncPostgresSaver:
        """Return the live AsyncPostgresSaver instance."""
        await self._ensure_langgraph_components()
        if self._checkpointer is None:
            raise RuntimeError("Database not initialized")
        return self._checkpointer

    async def get_store(self) -> AsyncPostgresStore:
        """Return the live AsyncPostgresStore instance."""
        await self._ensure_langgraph_components()
        if self._store is None:
            raise RuntimeError("Database not initialized")
        return self._store

    def get_engine(self) -> AsyncEngine:
        """Get the SQLAlchemy engine for metadata tables"""
        self.ensure_async_engine()
        if not self.engine:
            raise RuntimeError("Failed to initialize async engine")
        return self.engine

    def get_sync_engine(self) -> Engine:
        """Get the sync SQLAlchemy engine for sync code paths (e.g. Celery tasks)."""
        self.ensure_sync_engine()
        if not self.sync_engine:
            raise RuntimeError("Failed to initialize sync engine")
        return self.sync_engine


# Global database manager instance
db_manager = DatabaseManager()
