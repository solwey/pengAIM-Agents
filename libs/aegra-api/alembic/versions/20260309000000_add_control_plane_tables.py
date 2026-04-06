"""add_control_plane_tables

Revision ID: 606141af4b27
Revises: f539a8f934fb
Create Date: 2026-03-09 00:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision = "606141af4b27"
down_revision = "f539a8f934fb"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("duration_ms", sa.Integer, nullable=True))
    op.add_column("runs", sa.Column("current_step", sa.Text, nullable=True))

    op.create_table(
        "worker_heartbeat",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("status", sa.Text, nullable=False, server_default="online"),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_heartbeat",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("active_run_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "metadata",
            JSONB,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )

    op.create_table(
        "run_status_history",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.Text,
            sa.ForeignKey("runs.run_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("from_status", sa.Text, nullable=True),
        sa.Column("to_status", sa.Text, nullable=False),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("traceback", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "idx_run_status_history_run_id_created_at",
        "run_status_history",
        ["run_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_run_status_history_run_id_created_at", "run_status_history")
    op.drop_table("run_status_history")
    op.drop_table("worker_heartbeat")
    op.drop_column("runs", "current_step")
    op.drop_column("runs", "duration_ms")
