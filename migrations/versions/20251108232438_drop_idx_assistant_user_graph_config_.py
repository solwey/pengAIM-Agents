"""drop idx_assistant_user_graph_config index

Revision ID: 34ecbe1e8783
Revises: aee821a02fc8
Create Date: 2025-11-08 23:24:38.860549

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '34ecbe1e8783'
down_revision = 'aee821a02fc8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the idx_assistant_user_graph_config index
    op.drop_index("idx_assistant_user_graph_config", table_name="assistant")


def downgrade() -> None:
    # Recreate the idx_assistant_user_graph_config index
    op.create_index(
        "idx_assistant_user_graph_config",
        "assistant",
        ["user_id", "graph_id", "config"],
        unique=True,
    )
