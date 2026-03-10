"""add_run_metrics_columns

Revision ID: a1b2c3d4e5f6
Revises: 606141af4b27
Create Date: 2026-03-09 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "606141af4b27"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("tool_calls_count", sa.Integer, nullable=True))
    op.add_column("runs", sa.Column("tools_used", JSONB, nullable=True))
    op.add_column("runs", sa.Column("config_snapshot", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("runs", "config_snapshot")
    op.drop_column("runs", "tools_used")
    op.drop_column("runs", "tool_calls_count")
