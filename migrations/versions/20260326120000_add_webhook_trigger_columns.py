"""add webhook trigger columns to workflows

Revision ID: b3c4d5e6f7a8
Revises: a2794b43fa4b
Create Date: 2026-03-26 12:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "b3c4d5e6f7a8"
down_revision = "b3905c54gb5c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE workflows ADD COLUMN IF NOT EXISTS webhook_enabled BOOLEAN NOT NULL DEFAULT false"
    )
    op.execute(
        "ALTER TABLE workflows ADD COLUMN IF NOT EXISTS webhook_path TEXT"
    )
    op.execute(
        "ALTER TABLE workflows ADD COLUMN IF NOT EXISTS webhook_secret TEXT"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_webhook_path ON workflows (webhook_path)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_workflow_webhook_path")
    op.execute("ALTER TABLE workflows DROP COLUMN IF EXISTS webhook_secret")
    op.execute("ALTER TABLE workflows DROP COLUMN IF EXISTS webhook_path")
    op.execute("ALTER TABLE workflows DROP COLUMN IF EXISTS webhook_enabled")
