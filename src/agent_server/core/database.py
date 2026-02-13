"""Database manager with LangGraph integration"""
import asyncio
import os
from typing import Any

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import text

logger = structlog.get_logger(__name__)


class DatabaseManager:
    """Manages database connections and LangGraph persistence components"""

    def __init__(self) -> None:
        self.engine: AsyncEngine | None = None
        self._checkpointer: AsyncPostgresSaver | None = None
        self._checkpointer_cm: Any = None  # holds the contextmanager so we can close it
        self._store: AsyncPostgresStore | None = None
        self._store_cm: Any = None
        self._database_url = os.getenv(
            "DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/aegra"
        )
        self._langgraph_idle_ttl_seconds = int(os.getenv("LANGGRAPH_IDLE_TTL_SECONDS", "300"))
        self._langgraph_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize database connections and LangGraph components"""
        # SQLAlchemy for our minimal Agent Protocol metadata tables
        self.engine = create_async_engine(
            self._database_url,
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_pre_ping=True,
        )

        # Convert asyncpg URL to psycopg format for LangGraph
        # LangGraph packages require psycopg format, not asyncpg
        dsn = self._database_url.replace("postgresql+asyncpg://", "postgresql://")

        # Store connection string for creating LangGraph components on demand
        self._langgraph_dsn = dsn

        logger.info("✅ Database and LangGraph components initialized")

    async def close(self) -> None:
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()

        # Close the cached checkpointer if we opened one
        if self._checkpointer_cm is not None:
            await self._checkpointer_cm.__aexit__(None, None, None)
            self._checkpointer_cm = None
            self._checkpointer = None

        if self._store_cm is not None:
            await self._store_cm.__aexit__(None, None, None)
            self._store_cm = None
            self._store = None

        logger.info("✅ Database connections closed")

    async def _db_healthcheck(self) -> None:
        if self.engine is None:
            raise RuntimeError("Database not initialized")

        async with self.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    async def reset_langgraph_components(self) -> None:
        """Reset LangGraph components (checkpointer + store) on connection errors."""
        # Close existing LangGraph components safely
        if self._checkpointer_cm is not None:
            try:
                # Properly exit async context manager
                await self._checkpointer_cm.__aexit__(None, None, None)
            except Exception as e:
                # Log but do not raise: this is best-effort cleanup
                logger.warning("Error while closing checkpointer context", error=str(e))
            finally:
                self._checkpointer_cm = None
                self._checkpointer = None

        if self._store_cm is not None:
            try:
                await self._store_cm.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error while closing store context", error=str(e))
            finally:
                self._store_cm = None
                self._store = None

        logger.info("🔁 LangGraph components have been reset")

    async def get_checkpointer(self) -> AsyncPostgresSaver:
        """Return a live AsyncPostgresSaver.

        We enter the async context manager once and cache the saver so that
        subsequent calls reuse the same database connection pool.  LangGraph
        expects the *real* saver object (it calls methods like
        ``get_next_version``), so returning the context manager wrapper would
        fail.
        """
        if not hasattr(self, "_langgraph_dsn"):
            raise RuntimeError("Database not initialized")
        async with self._langgraph_lock:
            for attempt in range(2):
                if self._checkpointer is not None:
                    try:
                        await self._checkpointer.setup()
                        return self._checkpointer
                    except Exception as e:
                        msg = str(e)
                        if "closed" in msg.lower() or "connection" in msg.lower():
                            logger.warning(
                                "Checkpointer appears unhealthy; recycling LangGraph components",
                                error=msg,
                                attempt=attempt,
                            )
                            await self.reset_langgraph_components()
                        else:
                            raise

                try:
                    await self._db_healthcheck()
                    self._checkpointer_cm = AsyncPostgresSaver.from_conn_string(
                        self._langgraph_dsn
                    )
                    self._checkpointer = await self._checkpointer_cm.__aenter__()
                    # Ensure required tables exist (idempotent)
                    await self._checkpointer.setup()
                    return self._checkpointer
                except Exception as e:
                    msg = str(e)
                    await self.reset_langgraph_components()
                    if attempt == 0 and ("closed" in msg.lower() or "connection" in msg.lower()):
                        logger.warning(
                            "Failed to create checkpointer due to connection issue; retrying once",
                            error=msg,
                        )
                        continue
                    raise

            raise RuntimeError("Failed to create a healthy checkpointer")

    async def get_store(self) -> AsyncPostgresStore:
        """Return a live AsyncPostgresStore instance (vector + KV)."""
        if not hasattr(self, "_langgraph_dsn"):
            raise RuntimeError("Database not initialized")
        async with self._langgraph_lock:
            for attempt in range(2):
                if self._store is not None:
                    try:
                        await self._store.setup()
                        return self._store
                    except Exception as e:
                        msg = str(e)
                        if "closed" in msg.lower() or "connection" in msg.lower():
                            logger.warning(
                                "Store appears unhealthy; recycling LangGraph components",
                                error=msg,
                                attempt=attempt,
                            )
                            await self.reset_langgraph_components()
                        else:
                            raise

                try:
                    await self._db_healthcheck()
                    self._store_cm = AsyncPostgresStore.from_conn_string(self._langgraph_dsn)
                    self._store = await self._store_cm.__aenter__()
                    await self._store.setup()
                    return self._store
                except Exception as e:
                    msg = str(e)
                    await self.reset_langgraph_components()
                    if attempt == 0 and ("closed" in msg.lower() or "connection" in msg.lower()):
                        logger.warning(
                            "Failed to create store due to connection issue; retrying once",
                            error=msg,
                        )
                        continue
                    raise

            raise RuntimeError("Failed to create a healthy store")

    def get_engine(self) -> AsyncEngine:
        """Get the SQLAlchemy engine for metadata tables"""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        return self.engine


# Global database manager instance
db_manager = DatabaseManager()
