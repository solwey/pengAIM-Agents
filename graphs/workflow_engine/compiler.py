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
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph, add_messages

from graphs.workflow_engine.nodes import NODE_REGISTRY, build_condition_router
from graphs.workflow_engine.schema import NodeType, WorkflowDefinition

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State shared across all workflow nodes.

    - messages: standard LangGraph message list (for compatibility with SSE/streaming)
    - data: workflow-specific data dict where each node writes its results
    """

    messages: Annotated[list[AnyMessage], add_messages]
    data: dict[str, Any]


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
        node_fn = executor_cls.create(node_def.config)
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
