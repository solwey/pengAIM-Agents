"""Alembic environment configuration for Aegra database migrations."""

import asyncio
import threading
from logging.config import fileConfig

from sqlalchemy import pool, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Import your SQLAlchemy models here
from aegra_api.core.orm import Base
from aegra_api.settings import settings
from alembic import context

PROTOTYPE_SCHEMA = "prototype"

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Override the URL from settings — this respects DATABASE_URL, individual
# POSTGRES_* vars, and preserves query params (e.g. ?sslmode=require).
config.set_main_option("sqlalchemy.url", settings.db.database_url)

# Interpret the config file for Python logging.
# Only reconfigure logging when running from CLI (main thread).
# When invoked programmatically via asyncio.to_thread(), fileConfig()
# causes a cross-thread deadlock with the application's logging.
# See: https://github.com/sqlalchemy/alembic/discussions/1483
if config.config_file_name is not None and threading.current_thread() is threading.main_thread():
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata


def _include_object(object, name, type_, reflected, compare_to):
    """Exclude the tenants table and alembic_version from autogenerate."""
    if type_ == "table":
        if name == "alembic_version":
            return False
        return object.schema in [PROTOTYPE_SCHEMA, None]
    return True


def run_migrations_offline() -> None:
    raise NotImplementedError("Offline migrations are not supported")


async def _get_tenant_schemas() -> list[str]:
    """Return list of schemas to migrate: all enabled tenants + prototype.

    Uses a separate short-lived connection so the migration connection
    stays clean.
    """

    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url")

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        rows = await connection.execute(text("SELECT schema FROM public.tenants WHERE enabled = true"))
        schemas = [r[0] for r in rows.all()]
        if PROTOTYPE_SCHEMA not in schemas:
            schemas.append(PROTOTYPE_SCHEMA)
        return schemas


def do_run_migrations(connection: Connection, schemas: list[str]) -> None:
    """Run migrations once per tenant schema (plus the prototype schema)."""
    is_autogenerate = bool(context.get_x_argument(as_dictionary=True).get("autogenerate"))

    schemas = [PROTOTYPE_SCHEMA] if is_autogenerate else schemas
    for schema in schemas:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=False,
            include_object=_include_object,
            version_table_schema=schema,
        )

        connection.execute(text(f'SET search_path TO "{schema}", public'))

        with context.begin_transaction():
            context.run_migrations()

        print(f"Applied migrations to schema {schema}")


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    tenant_schemas = await _get_tenant_schemas()

    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url")

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations, tenant_schemas)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
