"""Entry point for the Dynamic Workflow graph.

This file is referenced by aegra.json as the graph source.
It exports a placeholder StateGraph that is registered with
the LangGraphService at startup. The actual workflow graph
is compiled dynamically from JSON via get_workflow_graph().

The placeholder ensures that:
1. The "Dynamic Workflow" graph appears in the registry
2. A default assistant is auto-created with a deterministic UUID
3. Clients can discover and reference the graph via the assistants API
"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph, add_messages


class WorkflowState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    data: dict[str, Any]


async def _placeholder_node(state: WorkflowState) -> dict:
    """Placeholder — real workflows are compiled dynamically."""
    return {
        "data": {
            "error": "This is a placeholder graph. "
            "Pass workflow_definition in config.configurable to run a real workflow."
        }
    }


builder = StateGraph(WorkflowState)
builder.add_node("placeholder", _placeholder_node)
builder.add_edge(START, "placeholder")
builder.add_edge("placeholder", END)

# Exported for aegra.json: "Dynamic Workflow": "./graphs/workflow_engine/entry.py:workflow_graph"
workflow_graph = builder
