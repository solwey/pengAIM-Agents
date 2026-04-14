"""Create a new tenant: insert into public.tenants and create the schema.

Usage:
    uv run --package aegra-api python scripts/create_tenant.py --schema acme
    uv run --package aegra-api python scripts/create_tenant.py --schema acme --disabled
    uv run --package aegra-api python scripts/create_tenant.py --schema acme --uuid 550e8400-e29b-41d4-a716-446655440000
"""

import argparse
import logging
import uuid
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from aegra_api.settings import settings

# Resolve alembic.ini / script_location relative to the aegra-api package so
# this script can be invoked from any CWD (e.g. the repo root).
_AEGRA_API_DIR = Path(__file__).resolve().parent.parent / "libs" / "aegra-api"
_ALEMBIC_INI = _AEGRA_API_DIR / "alembic.ini"
_ALEMBIC_SCRIPTS = _AEGRA_API_DIR / "alembic"


def _alembic_config() -> Config:
    cfg = Config(str(_ALEMBIC_INI), ini_section="alembic")
    cfg.set_main_option("script_location", str(_ALEMBIC_SCRIPTS))
    return cfg


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)-5.5s %(message)s")


def _sync_url() -> str:
    """Return a psycopg3-compatible sync SQLAlchemy URL."""
    return settings.db.database_url_sync.replace("postgresql://", "postgresql+psycopg://", 1)


def create_tenant(schema: str, enabled: bool = True, tenant_uuid: str | None = None) -> str:
    engine = create_engine(_sync_url())
    tenant_uuid = tenant_uuid or str(uuid.uuid4())

    with engine.connect() as conn:
        # Insert tenant row
        conn.execute(
            text("INSERT INTO public.tenants (uuid, schema, enabled) VALUES (:uuid, :schema, :enabled)"),
            {
                "uuid": tenant_uuid,
                "schema": schema,
                "enabled": enabled,
            },
        )
        # Create the schema
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        conn.commit()

    logger.info("Created tenant %s (schema=%s)", tenant_uuid, schema)

    # Run migrations for the new schema
    command.upgrade(_alembic_config(), "head")
    logger.info("Migrations applied for schema %s", schema)

    engine.dispose()
    return tenant_uuid


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new tenant")
    parser.add_argument("--schema", required=True, help="PostgreSQL schema name")
    parser.add_argument("--uuid", default=None, help="Explicit tenant UUID (auto-generated if omitted)")
    parser.add_argument("--disabled", action="store_true", help="Create tenant in disabled state")
    args = parser.parse_args()

    tenant_uuid = create_tenant(
        schema=args.schema,
        enabled=not args.disabled,
        tenant_uuid=args.uuid,
    )
    print(f"Tenant created: {tenant_uuid}")


if __name__ == "__main__":
    main()
