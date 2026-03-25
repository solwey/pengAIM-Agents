"""Switch node executor — multi-branch routing based on ordered conditions."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, compare, resolve_field
from graphs.workflow_engine.schema import SwitchConfig


class SwitchExecutor(NodeExecutor):
    """Switch nodes are pass-through — routing happens via conditional edges."""

    @staticmethod
    def create(config: dict[str, Any]):
        async def switch_node(
            state: dict[str, Any], config: RunnableConfig
        ) -> dict[str, Any]:
            return {}

        return switch_node


def build_switch_router(config: dict[str, Any]) -> callable:
    """Build a routing function that evaluates cases in order, returns first match label."""
    cfg = SwitchConfig(**config)

    def route(state: dict[str, Any]) -> str:
        data = state.get("data", {})
        for case in cfg.cases:
            actual = resolve_field(data, case.field)
            if compare(actual, case.operator.value, case.value):
                return case.label
        return cfg.default_label

    return route
