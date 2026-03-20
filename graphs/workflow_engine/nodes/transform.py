"""Transform node executor — merges key-value pairs into state["data"].

Supports {{template}} variables in string values, resolved from state["data"].
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_templates
from graphs.workflow_engine.schema import TransformConfig


def _resolve_values(values: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Resolve {{template}} variables in all string values recursively."""
    resolved: dict[str, Any] = {}
    for k, v in values.items():
        if isinstance(v, str):
            resolved[k] = resolve_templates(v, data)
        elif isinstance(v, dict):
            resolved[k] = _resolve_values(v, data)
        elif isinstance(v, list):
            resolved[k] = [
                resolve_templates(item, data) if isinstance(item, str) else item
                for item in v
            ]
        else:
            resolved[k] = v
    return resolved


class TransformExecutor(NodeExecutor):
    """Merge config-defined key-value pairs into state["data"]."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = TransformConfig(**config)

        async def transform_node(
            state: dict[str, Any], config: RunnableConfig
        ) -> dict[str, Any]:
            data = state.get("data", {})
            resolved_set = _resolve_values(cfg.set, data)
            return {"data": {**data, **resolved_set}}

        return transform_node
