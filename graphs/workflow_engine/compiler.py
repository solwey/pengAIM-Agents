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
import traceback
from collections.abc import Callable, Coroutine
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph, add_messages

from graphs.workflow_engine.nodes import (
    NODE_REGISTRY,
    build_condition_router,
    build_list_condition_router,
    build_source_condition_router,
    build_switch_router,
    build_tag_condition_router,
)
from graphs.workflow_engine.nodes.base import compare, resolve_field_strict
from graphs.workflow_engine.schema import ConditionConfig, NodeType, SwitchConfig, WorkflowDefinition

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
    has_error_edge: bool = False,
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
                found, actual = resolve_field_strict(data, cfg.field)
                if not found:
                    step["branch"] = "no"
                    step["field_missing"] = cfg.field
                else:
                    branch = "yes" if compare(actual, cfg.operator.value, cfg.value) else "no"
                    step["branch"] = branch
            elif node_type == "switch" and condition_config:
                sw_cfg = SwitchConfig(**condition_config)
                data = state.get("data", {})
                matched = sw_cfg.default_label
                missing_fields: list[str] = []
                for case in sw_cfg.cases:
                    found, actual = resolve_field_strict(data, case.field)
                    if not found:
                        missing_fields.append(case.field)
                        continue
                    if compare(actual, case.operator.value, case.value):
                        matched = case.label
                        break
                step["branch"] = matched
                if missing_fields:
                    step["field_missing"] = missing_fields
            elif "data" in result:
                current_data = state.get("data", {})
                changed = {k: v for k, v in result["data"].items() if k not in current_data or current_data[k] != v}
                if changed:
                    step["data"] = changed

            # Detect ok:False in node results and trigger on_error routing
            if has_error_edge and "data" in result:
                current_data = state.get("data", {})
                for key, val in result["data"].items():
                    if key not in current_data and isinstance(val, dict) and val.get("ok") is False:
                        error_msg = val.get("error", "Node returned ok: false")
                        full_msg = str(error_msg)
                        step["status"] = "failed"
                        step["error"] = full_msg[:500]
                        step["error_full"] = full_msg
                        result["data"]["_error"] = {
                            "node": node_id,
                            "message": full_msg[:500],
                            "full_message": full_msg,
                        }
                        break

            # Clear _error from previous nodes on success
            if step.get("status") != "failed":
                current_error = state.get("data", {}).get("_error")
                if current_error and current_error.get("node") != node_id:
                    if "data" not in result:
                        result["data"] = {**state.get("data", {})}
                    result["data"].pop("_error", None)

        except Exception as exc:
            full_exc_str = str(exc)
            tb_str = traceback.format_exc()
            step["status"] = "failed"
            step["error"] = full_exc_str[:500]
            step["error_full"] = full_exc_str
            step["error_traceback"] = tb_str

            if has_error_edge:
                # Don't raise — route to error handler via _error in state
                data = state.get("data", {})
                result = {
                    "data": {
                        **data,
                        "_error": {
                            "node": node_id,
                            "message": full_exc_str[:500],
                            "full_message": full_exc_str,
                            "traceback": tb_str,
                        },
                    }
                }
            else:
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

    # Collect disabled node IDs
    disabled_ids = {n.id for n in definition.nodes if not n.enabled}

    # Build a mapping: disabled node → its outgoing target (for rewiring).
    # For sequential edges: use the single outgoing target.
    # For disabled condition/switch nodes: pick a default branch so the graph
    # stays connected (switch → default_label, condition → "yes" branch),
    # falling back to the first defined branch.
    disabled_bypass: dict[str, str] = {}
    for node_id in disabled_ids:
        node_def = definition.get_node(node_id)
        target: str | None = None

        seq_edge = next(
            (e for e in definition.edges if e.from_node == node_id and e.type == "sequential"),
            None,
        )
        if seq_edge:
            target = seq_edge.to_node
        else:
            cond_edge = next(
                (
                    e
                    for e in definition.edges
                    if e.from_node == node_id and e.type in ("conditional", "switch") and e.branches
                ),
                None,
            )
            if cond_edge and cond_edge.branches:
                preferred: str | None = None
                if node_def and node_def.type == NodeType.SWITCH:
                    sw_cfg = SwitchConfig(**node_def.config)
                    preferred = cond_edge.branches.get(sw_cfg.default_label)
                    if sw_cfg.default_label not in cond_edge.branches:
                        logger.warning(
                            "Disabled switch node '%s' has no edge for default_label '%s'; "
                            "falling back to first available branch",
                            node_id,
                            sw_cfg.default_label,
                        )
                elif node_def and node_def.type in (
                    NodeType.CONDITION,
                    NodeType.TAG_CONDITION,
                    NodeType.LIST_CONDITION,
                    NodeType.SOURCE_CONDITION,
                ):
                    preferred = cond_edge.branches.get("yes")
                target = preferred or next(iter(cond_edge.branches.values()), None)

        if target is None:
            logger.warning(
                "Disabled node '%s' has no resolvable bypass target; routing to END",
                node_id,
            )
            target = "__end__"

        disabled_bypass[node_id] = target

    # Collect nodes that have on_error edges
    nodes_with_error_edges: set[str] = set()
    error_targets: dict[str, str] = {}  # node_id -> error_handler_node_id
    for edge in definition.edges:
        if edge.type == "on_error" and edge.to_node:
            nodes_with_error_edges.add(edge.from_node)
            error_targets[edge.from_node] = edge.to_node

    def _resolve_target(target: str | None) -> str | None:
        """Follow disabled node bypass chain."""
        seen: set[str] = set()
        while target and target in disabled_bypass and target not in seen:
            seen.add(target)
            target = disabled_bypass[target]
        return target

    # ── Register nodes ────────────────────────────────────────
    for node_def in definition.nodes:
        if node_def.id in disabled_ids:
            continue
        executor_cls = NODE_REGISTRY.get(node_def.type.value)
        if executor_cls is None:
            raise ValueError(
                f"No executor registered for node type '{node_def.type.value}'. "
                f"Available types: {list(NODE_REGISTRY.keys())}"
            )
        raw_fn = executor_cls.create(node_def.config)

        condition_config = None
        if node_def.type in (
            NodeType.CONDITION,
            NodeType.SWITCH,
            NodeType.TAG_CONDITION,
            NodeType.LIST_CONDITION,
            NodeType.SOURCE_CONDITION,
        ):
            condition_config = node_def.config

        node_fn = _wrap_with_step_tracking(
            raw_fn,
            node_def.id,
            node_def.type.value,
            condition_config=condition_config,
            has_error_edge=node_def.id in nodes_with_error_edges,
        )
        builder.add_node(node_def.id, node_fn)

    # ── Wire edges ────────────────────────────────────────────
    # Skip on_error edges in normal wiring — they are handled via error routing
    handled_by_error_routing: set[str] = set()

    for edge in definition.edges:
        if edge.type == "on_error":
            continue  # handled below

        # Skip edges originating from disabled nodes
        if edge.from_node in disabled_ids:
            continue

        src = START if edge.from_node == "__start__" else edge.from_node

        # If this node has an on_error edge AND a sequential edge,
        # we replace the sequential edge with conditional error routing
        if edge.type == "sequential" and edge.from_node in nodes_with_error_edges:
            if edge.from_node not in handled_by_error_routing:
                handled_by_error_routing.add(edge.from_node)

                resolved_to = _resolve_target(edge.to_node)
                normal_tgt = END if resolved_to == "__end__" else resolved_to
                error_tgt_id = error_targets[edge.from_node]
                resolved_err = _resolve_target(error_tgt_id)
                error_tgt = END if resolved_err == "__end__" else resolved_err

                def _make_error_router(node_id: str):
                    def route(state: dict[str, Any]) -> str:
                        err = state.get("data", {}).get("_error")
                        if err and isinstance(err, dict) and err.get("node") == node_id:
                            return "__error__"
                        return "__normal__"

                    return route

                builder.add_conditional_edges(
                    src,
                    _make_error_router(edge.from_node),
                    {"__normal__": normal_tgt, "__error__": error_tgt},
                )
            continue

        if edge.type == "sequential":
            resolved_to = _resolve_target(edge.to_node)
            tgt = END if resolved_to == "__end__" else resolved_to
            if tgt is not None:
                builder.add_edge(src, tgt)

        elif edge.type == "conditional":
            condition_node = definition.get_node(edge.from_node)
            if condition_node is None:
                raise ValueError(f"Conditional edge references non-existent node: '{edge.from_node}'")

            if condition_node.type == NodeType.TAG_CONDITION:
                route_fn = build_tag_condition_router(condition_node.config)
            elif condition_node.type == NodeType.LIST_CONDITION:
                route_fn = build_list_condition_router(condition_node.config)
            elif condition_node.type == NodeType.SOURCE_CONDITION:
                route_fn = build_source_condition_router(condition_node.config)
            else:
                route_fn = build_condition_router(condition_node.config)

            mapping: dict[str, str] = {}
            for label, target in (edge.branches or {}).items():
                resolved = _resolve_target(target)
                if resolved is not None:
                    mapping[label] = END if resolved == "__end__" else resolved

            builder.add_conditional_edges(src, route_fn, mapping)

        elif edge.type == "switch":
            switch_node = definition.get_node(edge.from_node)
            if switch_node is None:
                raise ValueError(f"Switch edge references non-existent node: '{edge.from_node}'")

            route_fn = build_switch_router(switch_node.config)

            mapping = {}
            for label, target in (edge.branches or {}).items():
                resolved = _resolve_target(target)
                if resolved is not None:
                    mapping[label] = END if resolved == "__end__" else resolved

            builder.add_conditional_edges(src, route_fn, mapping)

    logger.info(
        "Compiled workflow '%s' with %d nodes and %d edges",
        definition.name,
        len(definition.nodes),
        len(definition.edges),
    )

    return builder
