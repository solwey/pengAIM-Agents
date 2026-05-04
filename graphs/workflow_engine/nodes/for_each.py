"""For-each loop node — runs a child workflow once per item in a list.

Reads ``state["data"][items_key]`` (dot-path), iterates each item, and runs
the child workflow per-item with the item exposed under ``item_var`` in the
child's data. Outputs are aggregated under ``response_key``.

Cycle and depth protection mirror sub_workflow — the parent workflow_chain
is propagated, and the child's id is appended for nested-loop detection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from sqlalchemy import select

from aegra_api.core.orm import Workflow, new_tenant_sync_session
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import ForEachConfig, WorkflowDefinition

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


class ForEachExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = ForEachConfig(**config)

        async def for_each_node(state: dict, config: RunnableConfig) -> dict:
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
                logger.warning("for_each cycle blocked: %s", cycle)
                return _fail(f"for_each cycle detected: {cycle}")
            if len(chain) >= cfg.max_depth:
                return _fail(f"for_each max_depth={cfg.max_depth} exceeded (chain length {len(chain)})")

            items = resolve_field(data, cfg.items_key)
            if items is None:
                return _fail(f"items_key '{cfg.items_key}' not found in state")
            if not isinstance(items, list):
                return _fail(f"items_key '{cfg.items_key}' must resolve to a list, got {type(items).__name__}")
            if len(items) > cfg.max_items:
                return _fail(f"for_each: {len(items)} items exceeds max_items={cfg.max_items}")

            definition_dict = await asyncio.to_thread(_load_definition_sync, tenant_schema, cfg.workflow_id)
            if definition_dict is None:
                return _fail(f"Workflow '{cfg.workflow_id}' not found or inactive")

            try:
                definition = WorkflowDefinition(**definition_dict)
                graph = compile_workflow(definition)
                compiled = graph.compile()
            except Exception as exc:  # noqa: BLE001
                logger.warning("for_each compile failed: %s", exc)
                return _fail(f"for_each compile failed: {exc}")

            child_configurable = {**configurable, "workflow_chain": [*chain, cfg.workflow_id]}
            semaphore = asyncio.Semaphore(cfg.concurrency)

            async def _run_one(idx: int, item: Any) -> dict[str, Any]:
                async with semaphore:
                    child_state = {
                        "messages": [],
                        "data": {cfg.item_var: item, cfg.index_var: idx},
                        "steps": [],
                    }
                    try:
                        out = await compiled.ainvoke(child_state, config={"configurable": child_configurable})
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("for_each item %d failed: %s", idx, exc)
                        return {"ok": False, "index": idx, "error": str(exc)[:500]}
                    return {"ok": True, "index": idx, "data": out.get("data", {})}

            outputs: list[dict[str, Any]] = []
            failures = 0
            if cfg.fail_fast:
                for idx, item in enumerate(items):
                    res = await _run_one(idx, item)
                    outputs.append(res)
                    if not res.get("ok"):
                        failures += 1
                        break
            else:
                tasks = [asyncio.create_task(_run_one(idx, item)) for idx, item in enumerate(items)]
                outputs = await asyncio.gather(*tasks)
                failures = sum(1 for r in outputs if not r.get("ok"))

            logger.info(
                "for_each complete: %d items, %d ok, %d failed (concurrency=%d)",
                len(items),
                len(outputs) - failures,
                failures,
                cfg.concurrency,
            )

            return {
                "data": {
                    **data,
                    cfg.response_key: {
                        "ok": failures == 0,
                        "total": len(items),
                        "succeeded": len(outputs) - failures,
                        "failed": failures,
                        "outputs": outputs,
                    },
                }
            }

        return for_each_node
