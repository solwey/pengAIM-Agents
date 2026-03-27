"""add webhook trigger columns to workflows

Revision ID: b3c4d5e6f7a8
Revises: b3905c54gb5c
Create Date: 2026-03-26 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "b3c4d5e6f7a8"
down_revision = "b3905c54gb5c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "workflows",
        sa.Column(
            "webhook_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column("workflows", sa.Column("webhook_path", sa.Text(), nullable=True))
    op.add_column("workflows", sa.Column("webhook_secret", sa.Text(), nullable=True))
    op.create_index(
        "idx_workflow_webhook_path", "workflows", ["webhook_path"], unique=True
    )


def downgrade() -> None:
    op.drop_index("idx_workflow_webhook_path", table_name="workflows")
    op.drop_column("workflows", "webhook_secret")
    op.drop_column("workflows", "webhook_path")
    op.drop_column("workflows", "webhook_enabled")
