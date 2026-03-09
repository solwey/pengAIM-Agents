"""add_run_metrics_columns

Revision ID: a1b2c3d4e5f6
Revises: 606141af4b27
Create Date: 2026-03-09 10:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "606141af4b27"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Use raw SQL with IF NOT EXISTS for idempotency (safe for prod)
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS tool_calls_count INTEGER")
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS tools_used JSONB")
    op.execute("ALTER TABLE runs ADD COLUMN IF NOT EXISTS config_snapshot JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS config_snapshot")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS tools_used")
    op.execute("ALTER TABLE runs DROP COLUMN IF EXISTS tool_calls_count")
