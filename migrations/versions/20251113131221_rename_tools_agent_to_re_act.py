"""rename_tools_agent_to_re-act

Revision ID: c4f4fd4e6fb5
Revises: 5b125b55a942
Create Date: 2025-11-13 13:12:21.006709

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'c4f4fd4e6fb5'
down_revision = '5b125b55a942'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "UPDATE assistant "
        "SET graph_id = REPLACE(graph_id, 'Tools Agent', 'ReAct Agent'), "
        "    name = REPLACE(name, 'Tools Agent', 'ReAct Agent'), "
        "    description = REPLACE(description, 'Tools Agent', 'ReAct Agent') "
        "WHERE graph_id LIKE '%Tools Agent%' "
        "   OR name LIKE '%Tools Agent%' "
        "   OR description LIKE '%Tools Agent%';"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE assistant "
        "SET graph_id = REPLACE(graph_id, 'ReAct Agent', 'Tools Agent'), "
        "    name = REPLACE(name, 'ReAct Agent', 'Tools Agent'), "
        "    description = REPLACE(description, 'ReAct Agent', 'Tools Agent') "
        "WHERE graph_id LIKE '%ReAct Agent%' "
        "   OR name LIKE '%ReAct Agent%' "
        "   OR description LIKE '%ReAct Agent%';"
    )
