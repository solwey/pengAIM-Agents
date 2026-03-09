"""add_control_plane_tables

Revision ID: 606141af4b27
Revises: f539a8f934fb
Create Date: 2026-03-09 00:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "606141af4b27"
down_revision = "f539a8f934fb"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Use raw SQL with IF NOT EXISTS for idempotency (safe for prod)
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS duration_ms INTEGER")
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS current_step TEXT")

    op.execute("""
        CREATE TABLE IF NOT EXISTS worker_heartbeat (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'online',
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_heartbeat TIMESTAMPTZ NOT NULL DEFAULT now(),
            active_run_count INTEGER NOT NULL DEFAULT 0,
            metadata JSONB DEFAULT '{}'::jsonb
        )
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS run_status_history (
            id SERIAL PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            from_status TEXT,
            to_status TEXT NOT NULL,
            error_message TEXT,
            traceback TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_run_status_history_run_id_created_at
        ON run_status_history (run_id, created_at)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_run_status_history_run_id_created_at")
    op.execute("DROP TABLE IF EXISTS run_status_history")
    op.execute("DROP TABLE IF EXISTS worker_heartbeat")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS current_step")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS duration_ms")
