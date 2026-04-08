"""Add execution_params and lease columns to runs table

Supports the worker executor architecture:
- execution_params: JSONB storing RunJob serialization so workers can
  reconstruct the job from the database after receiving a run_id via Redis.
- claimed_by: identifies which worker owns a run (lease holder).
- lease_expires_at: when the lease expires; a reaper re-enqueues runs
  whose leases have expired (worker crashed).

Revision ID: 84872197912a
Revises: b536da7e15d9
Create Date: 2026-04-08 03:35:30.158233

"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision = "84872197912a"
down_revision = "b536da7e15d9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("execution_params", JSONB(), nullable=True))
    op.add_column("runs", sa.Column("claimed_by", sa.Text(), nullable=True))
    op.add_column(
        "runs",
        sa.Column("lease_expires_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.create_index("idx_runs_lease_reaper", "runs", ["status", "lease_expires_at"])


def downgrade() -> None:
    op.drop_index("idx_runs_lease_reaper", table_name="runs")
    op.drop_column("runs", "lease_expires_at")
    op.drop_column("runs", "claimed_by")
    op.drop_column("runs", "execution_params")
