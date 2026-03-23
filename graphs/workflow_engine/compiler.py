"""Workflow compiler — converts a validated JSON definition into a LangGraph StateGraph.

The compiler:
1. Validates the JSON via WorkflowDefinition (Pydantic)
2. Builds a WorkflowState TypedDict
3. Creates async node functions via NODE_REGISTRY
4. Wires edges (sequential + conditional)
5. Returns an UNCOMPILED StateGraph — LangGraphService compiles it with checkpointer
"""

from __future__ import annotations

import logging
import operator
import time
from collections.abc import Callable, Coroutine
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages

from graphs.workflow_engine.nodes import NODE_REGISTRY, build_condition_router
from graphs.workflow_engine.nodes.base import compare, resolve_field
from graphs.workflow_engine.schema import ConditionConfig, NodeType, WorkflowDefinition

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State shared across all workflow nodes.

    - messages: standard LangGraph message list (for compatibility with SSE/streaming)
    - data: workflow-specific data dict where each node writes its results
    - steps: ordered list of executed steps, auto-appended by node wrapper
    """

    messages: Annotated[list[AnyMessage], add_messages]
    data: dict[str, Any]
    steps: Annotated[list[dict[str, Any]], operator.add]


def _wrap_with_step_tracking(
    fn: Callable[..., Coroutine[Any, Any, dict]],
    node_id: str,
    node_type: str,
    condition_config: dict[str, Any] | None = None,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Wrap a node function to record an executed step in state["steps"]."""

    async def wrapped(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        start = time.monotonic()
        step: dict[str, Any] = {"node": node_id, "type": node_type}

        try:
            result = await fn(state, config)
            step["status"] = "completed"

            if node_type == "condition" and condition_config:
                cfg = ConditionConfig(**condition_config)
                data = state.get("data", {})
                actual = resolve_field(data, cfg.field)
                branch = (
                    "yes"
                    if compare(actual, cfg.operator.value, cfg.value)
                    else "no"
                )
                step["branch"] = branch
            elif "data" in result:
                current_data = state.get("data", {})
                changed = {
                    k: v
                    for k, v in result["data"].items()
                    if k not in current_data or current_data[k] != v
                }
                if changed:
                    step["data"] = changed

        except Exception as exc:
            step["status"] = "failed"
            step["error"] = str(exc)[:500]
            raise

        finally:
            step["duration_ms"] = round((time.monotonic() - start) * 1000)

        result["steps"] = [step]
        return result

    return wrapped


def compile_workflow(definition: WorkflowDefinition) -> StateGraph:
    """Compile a WorkflowDefinition into an uncompiled LangGraph StateGraph.

    The returned StateGraph is NOT compiled — the caller (LangGraphService)
    is responsible for compiling with the PostgreSQL checkpointer.
    """
    builder = StateGraph(WorkflowState)

    # ── Register nodes ────────────────────────────────────────
    for node_def in definition.nodes:
        executor_cls = NODE_REGISTRY.get(node_def.type.value)
        if executor_cls is None:
            raise ValueError(
                f"No executor registered for node type '{node_def.type.value}'. "
                f"Available types: {list(NODE_REGISTRY.keys())}"
            )
        raw_fn = executor_cls.create(node_def.config)
        node_fn = _wrap_with_step_tracking(
            raw_fn,
            node_def.id,
            node_def.type.value,
            condition_config=node_def.config
            if node_def.type == NodeType.CONDITION
            else None,
        )
        builder.add_node(node_def.id, node_fn)

    # ── Wire edges ────────────────────────────────────────────
    for edge in definition.edges:
        src = START if edge.from_node == "__start__" else edge.from_node

        if edge.type == "sequential":
            tgt = END if edge.to_node == "__end__" else edge.to_node
            builder.add_edge(src, tgt)

        elif edge.type == "conditional":
            # Get the condition node's config for building the router
            condition_node = definition.get_node(edge.from_node)
            if condition_node is None:
                raise ValueError(
                    f"Conditional edge references non-existent node: '{edge.from_node}'"
                )

            route_fn = build_condition_router(condition_node.config)

            # Build the mapping: "yes" -> target_node, "no" -> target_node
            mapping: dict[str, str] = {}
            for label, target in (edge.branches or {}).items():
                mapping[label] = END if target == "__end__" else target

            builder.add_conditional_edges(src, route_fn, mapping)

    logger.info(
        "Compiled workflow '%s' with %d nodes and %d edges",
        definition.name,
        len(definition.nodes),
        len(definition.edges),
    )

    return builder
