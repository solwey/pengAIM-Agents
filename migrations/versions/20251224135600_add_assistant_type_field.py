"""add_assistant_type_field

Revision ID: add_type_field
Revises: 
Create Date: 2025-12-24 13:56:00.314349

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "add_type_field"
down_revision = "d2978cc4bf58"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add type column to assistant table
    op.add_column(
        "assistant", sa.Column("type", sa.Text(), nullable=True)
    )
    op.create_index(
        "idx_assistant_type", "assistant", ["type"], unique=False
    )


def downgrade() -> None:
    # Drop index and column
    op.drop_index("idx_assistant_type", table_name="assistant")
    op.drop_column("assistant", "type")
