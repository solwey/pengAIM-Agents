"""add workflow_schedules table

Revision ID: b3905c54gb5c
Revises: a2794b43fa4b
Create Date: 2026-03-25 10:47:31.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'b3905c54gb5c'
down_revision = 'a2794b43fa4b'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('workflow_schedules',
    sa.Column('id', sa.Text(), server_default=sa.text('uuid_generate_v4()::text'), nullable=False),
    sa.Column('workflow_id', sa.Text(), nullable=False),
    sa.Column('team_id', sa.Text(), nullable=False),
    sa.Column('user_id', sa.Text(), nullable=False),
    sa.Column('cron_expression', sa.Text(), nullable=False),
    sa.Column('timezone', sa.Text(), server_default=sa.text("'UTC'"), nullable=False),
    sa.Column('is_enabled', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('input_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('last_run_at', sa.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('next_run_at', sa.TIMESTAMP(timezone=True), nullable=True),
    sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['workflow_id'], ['workflows.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_schedule_workflow', 'workflow_schedules', ['workflow_id'], unique=False)
    op.create_index('idx_schedule_team', 'workflow_schedules', ['team_id'], unique=False)
    op.create_index('idx_schedule_enabled_next', 'workflow_schedules', ['is_enabled', 'next_run_at'], unique=False)
    op.create_index(
        'idx_schedule_unique_active_workflow',
        'workflow_schedules',
        ['workflow_id'],
        unique=True,
        postgresql_where=sa.text('is_enabled = true'),
    )


def downgrade() -> None:
    op.drop_index('idx_schedule_unique_active_workflow', table_name='workflow_schedules')
    op.drop_index('idx_schedule_enabled_next', table_name='workflow_schedules')
    op.drop_index('idx_schedule_team', table_name='workflow_schedules')
    op.drop_index('idx_schedule_workflow', table_name='workflow_schedules')
    op.drop_table('workflow_schedules')
