"""Sub-workflow node — invokes another workflow as a step.

Loads the child WorkflowDefinition by id from the same tenant schema,
compiles it, and runs it inline with the parent's configurable so that
team_id / auth_token / ingestion config propagate naturally.

Cycle protection: each invocation appends its workflow_id to
``configurable["workflow_chain"]``; the executor refuses to run a child
already present in the chain.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from sqlalchemy import select

from aegra_api.core.orm import Workflow, new_tenant_sync_session
from graphs.workflow_engine.nodes.base import NodeExecutor
from graphs.workflow_engine.schema import SubWorkflowConfig, WorkflowDefinition

logger = logging.getLogger(__name__)


def _load_definition_sync(tenant_schema: str, workflow_id: str) -> dict[str, Any] | None:
    with new_tenant_sync_session(tenant_schema) as session:
        row = session.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.is_active.is_(True),
                Workflow.deleted_at.is_(None),
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return dict(row.definition)


class SubWorkflowExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = SubWorkflowConfig(**config)

        async def sub_workflow_node(state: dict, config: RunnableConfig) -> dict:
            from graphs.workflow_engine.compiler import compile_workflow

            data = state.get("data", {})
            configurable = config.get("configurable", {})
            tenant_schema = configurable.get("tenant_schema", "")
            chain: list[str] = list(configurable.get("workflow_chain", []))

            def _fail(error: str) -> dict:
                return {"data": {**data, cfg.response_key: {"ok": False, "error": error}}}

            if not tenant_schema:
                return _fail("tenant_schema missing from configurable")

            if cfg.workflow_id in chain:
                cycle = " -> ".join([*chain, cfg.workflow_id])
                logger.warning("Sub-workflow cycle blocked: %s", cycle)
                return _fail(f"Sub-workflow cycle detected: {cycle}")

            definition_dict = await asyncio.to_thread(_load_definition_sync, tenant_schema, cfg.workflow_id)
            if definition_dict is None:
                return _fail(f"Workflow '{cfg.workflow_id}' not found or inactive")

            try:
                definition = WorkflowDefinition(**definition_dict)
                graph = compile_workflow(definition)
                compiled = graph.compile()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sub-workflow compile failed: %s", exc)
                return _fail(f"Sub-workflow compile failed: {exc}")

            child_configurable = {**configurable, "workflow_chain": [*chain, cfg.workflow_id]}
            child_state = {"messages": [], "data": dict(data), "steps": []}

            try:
                child_result = await compiled.ainvoke(child_state, config={"configurable": child_configurable})
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sub-workflow execution failed: %s", exc)
                return _fail(f"Sub-workflow execution failed: {exc}")

            child_data = child_result.get("data", {})
            return {
                "data": {
                    **data,
                    cfg.response_key: {
                        "ok": True,
                        "data": child_data,
                        "steps": child_result.get("steps", []),
                    },
                }
            }

        return sub_workflow_node
