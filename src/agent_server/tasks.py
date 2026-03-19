"""Celery tasks for sweep/cleanup operations and workflow execution."""

import asyncio
import os
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import create_engine, delete, select, update
from sqlalchemy.orm import Session

from .celery_app import celery_app
from .core.orm import Run as RunORM
from .core.orm import RunStatusHistory, Workflow, WorkflowRun, WorkerHeartbeat
from graphs.workflow_engine.compiler import compile_workflow
from graphs.workflow_engine.schema import WorkflowDefinition

logger = structlog.getLogger(__name__)

HEARTBEAT_TIMEOUT_SECONDS = 45
ZOMBIE_RUN_TIMEOUT_MINUTES = 30


def _get_sync_engine():
    """Create a sync SQLAlchemy engine for Celery tasks."""
    url = os.getenv(
        "DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/aegra"
    )
    # Convert async URL to sync
    sync_url = url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
    return create_engine(sync_url)


@celery_app.task(name="src.agent_server.tasks.sweep_stale_workers")
def sweep_stale_workers():
    """Mark workers as offline if their heartbeat is stale (>45s)."""
    engine = _get_sync_engine()
    cutoff = datetime.now(UTC) - timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS)

    with Session(engine) as session:
        result = session.execute(
            update(WorkerHeartbeat)
            .where(
                WorkerHeartbeat.last_heartbeat < cutoff,
                WorkerHeartbeat.status != "offline",
            )
            .values(status="offline")
        )
        session.commit()
        marked = result.rowcount

    if marked:
        logger.info("sweep_stale_workers", marked_offline=marked)
    return {"marked_offline": marked}


@celery_app.task(name="src.agent_server.tasks.sweep_zombie_runs")
def sweep_zombie_runs():
    """Mark runs stuck in active status for 30+ minutes as failed."""
    engine = _get_sync_engine()
    cutoff = datetime.now(UTC) - timedelta(minutes=ZOMBIE_RUN_TIMEOUT_MINUTES)
    error_msg = "Zombie run detected: no activity for 30+ minutes"

    with Session(engine) as session:
        # Find affected runs
        stmt = select(RunORM).where(
            RunORM.status.in_(["pending", "running", "streaming"]),
            RunORM.updated_at < cutoff,
        )
        affected_runs = session.scalars(stmt).all()

        if not affected_runs:
            return {"marked_failed": 0}

        # Insert status history for each affected run
        for run in affected_runs:
            history = RunStatusHistory(
                run_id=run.run_id,
                from_status=run.status,
                to_status="failed",
                error_message=error_msg,
                created_at=datetime.now(UTC),
            )
            session.add(history)

        # Bulk update runs to failed
        update_stmt = (
            update(RunORM)
            .where(
                RunORM.status.in_(["pending", "running", "streaming"]),
                RunORM.updated_at < cutoff,
            )
            .values(status="failed", error_message=error_msg)
        )
        result = session.execute(update_stmt)
        session.commit()

        zombie_run_ids = [r.run_id for r in affected_runs]
        marked = result.rowcount
        logger.warning(
            "sweep_zombie_runs", marked_failed=marked, run_ids=zombie_run_ids
        )
        return {"marked_failed": marked}


@celery_app.task(name="src.agent_server.tasks.cleanup_offline_workers")
def cleanup_offline_workers(max_age_hours: int = 0):
    """Delete offline workers. Use max_age_hours to keep recent ones (0 = all)."""
    engine = _get_sync_engine()

    with Session(engine) as session:
        conditions = [WorkerHeartbeat.status == "offline"]
        if max_age_hours > 0:
            cutoff = datetime.now(UTC) - timedelta(hours=max_age_hours)
            conditions.append(WorkerHeartbeat.last_heartbeat < cutoff)
        result = session.execute(delete(WorkerHeartbeat).where(*conditions))
        session.commit()
        removed = result.rowcount

    if removed:
        logger.info("cleanup_offline_workers", removed=removed)
    return {"removed": removed}


@celery_app.task(
    name="src.agent_server.tasks.execute_workflow",
    bind=True,
    max_retries=0,
)
def execute_workflow(self, workflow_run_id: str):
    """Execute a workflow run: compile JSON → LangGraph → ainvoke.

    Steps:
    1. Load WorkflowRun from DB, set status=running
    2. Load Workflow definition
    3. Validate and compile to StateGraph
    4. Execute via ainvoke (no streaming, no checkpointer)
    5. Save output or error to WorkflowRun
    """
    engine = _get_sync_engine()

    with Session(engine) as session:
        run = session.execute(
            select(WorkflowRun).where(WorkflowRun.id == workflow_run_id)
        ).scalar_one_or_none()

        if not run:
            logger.error("execute_workflow: run not found", run_id=workflow_run_id)
            return {"error": "run not found"}

        if run.status == "cancelled":
            logger.info("execute_workflow: run already cancelled", run_id=workflow_run_id)
            return {"status": "cancelled"}

        run.status = "running"
        run.started_at = datetime.now(UTC)
        session.commit()

        try:
            workflow = session.execute(
                select(Workflow).where(Workflow.id == run.workflow_id)
            ).scalar_one_or_none()

            if not workflow:
                raise ValueError(f"Workflow {run.workflow_id} not found")

            definition = WorkflowDefinition(**workflow.definition)
            graph = compile_workflow(definition)
            compiled = graph.compile()

            initial_state = {
                "messages": [],
                "data": run.input_data or {},
            }

            result = asyncio.run(compiled.ainvoke(initial_state))

            final_data = result.get("data", {})

            # Reconstruct execution steps from definition + final state
            steps = []
            skipped_nodes = set()

            # Find which branch was skipped
            for edge in definition.edges:
                if edge.type == "conditional" and edge.branches:
                    branch = "yes" if final_data.get("status") == "success" else "no"
                    skipped = "no" if branch == "yes" else "yes"
                    skipped_node = edge.branches.get(skipped)
                    if skipped_node and skipped_node != "__end__":
                        skipped_nodes.add(skipped_node)

            for node_def in definition.nodes:
                if node_def.id in skipped_nodes:
                    continue
                step: dict = {"node": node_def.id, "type": node_def.type.value}
                if node_def.type.value == "api_request":
                    resp_key = node_def.config.get("response_key", "api_response")
                    step["data"] = {resp_key: final_data.get(resp_key)}
                elif node_def.type.value == "condition":
                    step["branch"] = "yes" if final_data.get("status") == "success" else "no"
                elif node_def.type.value == "transform":
                    step["data"] = {
                        k: final_data.get(k)
                        for k in node_def.config.get("set", {}).keys()
                        if k in final_data
                    }
                steps.append(step)

            run.output_data = {**final_data, "steps": steps}
            run.status = "completed"
            run.completed_at = datetime.now(UTC)
            session.commit()

            logger.info(
                "execute_workflow: completed",
                run_id=workflow_run_id,
                workflow_name=definition.name,
            )
            return {"status": "completed", "run_id": workflow_run_id}

        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)[:2000]
            run.completed_at = datetime.now(UTC)
            session.commit()

            logger.error(
                "execute_workflow: failed",
                run_id=workflow_run_id,
                error=str(exc),
            )
            return {"status": "failed", "error": str(exc)}
