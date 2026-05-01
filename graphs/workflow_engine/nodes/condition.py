"""Condition node executor — pass-through node + router builder for branching."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import (
    NodeExecutor,
    compare,
    resolve_field_strict,
)
from graphs.workflow_engine.schema import ConditionConfig

logger = logging.getLogger(__name__)


class ConditionExecutor(NodeExecutor):
    """Condition nodes are pass-through — routing happens via conditional edges."""

    @staticmethod
    def create(_config: dict[str, Any]):
        async def condition_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            # No state mutation — routing is handled by the edge router
            return {}

        return condition_node


def build_condition_router(
    config: dict[str, Any],
) -> Callable[..., str]:
    """Build a routing function for add_conditional_edges().

    Returns a function that reads state["data"], evaluates the condition,
    and returns "yes" or "no".
    """
    cfg = ConditionConfig(**config)

    def route(state: dict[str, Any]) -> Literal["yes", "no"]:
        data = state.get("data", {})
        found, actual = resolve_field_strict(data, cfg.field)
        if not found:
            logger.warning("Condition field '%s' missing from state.data — routing to 'no'", cfg.field)
            return "no"
        result = compare(actual, cfg.operator.value, cfg.value)
        return "yes" if result else "no"

    return route
