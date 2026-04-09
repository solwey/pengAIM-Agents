import logging
import time
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from aegra_api.settings import settings
from alembic import command
from alembic.config import Config

# Resolve alembic.ini / script_location relative to the aegra-api package so
# these scripts can be invoked from any CWD (e.g. the repo root).
_AEGRA_API_DIR = Path(__file__).resolve().parent.parent / "libs" / "aegra-api"
_ALEMBIC_INI = _AEGRA_API_DIR / "alembic.ini"
_ALEMBIC_SCRIPTS = _AEGRA_API_DIR / "alembic"


def _alembic_config() -> Config:
    cfg = Config(str(_ALEMBIC_INI), ini_section="alembic")
    cfg.set_main_option("script_location", str(_ALEMBIC_SCRIPTS))
    return cfg

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PROTOTYPE_SCHEMA = "prototype"

TENANTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS public.tenants (
    uuid        VARCHAR(36) PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    schema      VARCHAR(63) NOT NULL,
    kc_realm    VARCHAR(255) NOT NULL,
    enabled     BOOLEAN NOT NULL DEFAULT true,
    CONSTRAINT uq_tenant_schema   UNIQUE (schema),
    CONSTRAINT uq_tenant_kc_realm UNIQUE (kc_realm)
);
"""


def _sync_url() -> str:
    """Return a psycopg3-compatible sync SQLAlchemy URL."""
    return settings.db.database_url_sync.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )


def create_database_if_not_exist() -> None:
    """Create the target database if it doesn't already exist."""
    db_name = settings.db.POSTGRES_DB
    maintenance_url = _sync_url().rsplit("/", 1)[0] + "/postgres"
    engine = create_engine(maintenance_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db"), {"db": db_name}
        )
        if not result.scalar():
            logger.info("Creating database %s", db_name)
            conn.execute(text(f'CREATE DATABASE "{db_name}"'))
        else:
            logger.info("Database %s already exists", db_name)
    engine.dispose()


def ensure_tenants_table() -> None:
    """Create the public.tenants table if it doesn't exist."""
    engine = create_engine(_sync_url())
    with engine.connect() as conn:
        conn.execute(text(TENANTS_TABLE_DDL))
        conn.commit()
    engine.dispose()
    logger.info("Ensured public.tenants table exists")


def ensure_prototype_schema() -> None:
    """Create the prototype schema if it doesn't exist."""
    engine = create_engine(_sync_url())
    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {PROTOTYPE_SCHEMA}"))
        conn.commit()
    engine.dispose()
    logger.info("Ensured %s schema exists", PROTOTYPE_SCHEMA)


def ensure_tenant_schemas() -> None:
    """Create schemas for all enabled tenants that don't exist yet."""
    engine = create_engine(_sync_url())
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT schema FROM public.tenants WHERE enabled = true")
        ).fetchall()
        for (schema_name,) in rows:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))
        conn.commit()
    engine.dispose()
    logger.info("Ensured all tenant schemas exist")


def run_migrations() -> None:
    create_database_if_not_exist()
    ensure_tenants_table()
    ensure_prototype_schema()
    ensure_tenant_schemas()

    command.upgrade(_alembic_config(), "head")


def main() -> dict[str, int]:
    try:
        for attempt in range(3):
            try:
                run_migrations()
                break
            except OperationalError:
                if attempt < 2:
                    logger.warning(
                        "Attempt %d failed for database; retrying in 5 seconds...",
                        attempt + 1,
                    )
                    time.sleep(5)
                else:
                    raise Exception(
                        f"Attempt {attempt + 1} failed; no more retries for database"
                    )

        logger.info("All migrations applied")
        return {"statusCode": 200}
    except Exception as e:
        logger.error("Migration error: %s", e)
        return {"statusCode": 500}


if __name__ == "__main__":
    main()
