"""Transform node executor — merges static key-value pairs into state["data"]."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor
from graphs.workflow_engine.schema import TransformConfig


class TransformExecutor(NodeExecutor):
    """Merge config-defined key-value pairs into state["data"]."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = TransformConfig(**config)

        async def transform_node(
            state: dict[str, Any], config: RunnableConfig
        ) -> dict[str, Any]:
            data = state.get("data", {})
            return {"data": {**data, **cfg.set}}

        return transform_node
