"""Delay node executor — pauses workflow execution for a configured duration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor
from graphs.workflow_engine.schema import DelayConfig

logger = logging.getLogger(__name__)


class DelayExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = DelayConfig(**config)

        async def delay_node(state: dict, config: RunnableConfig) -> dict:
            logger.info("Delay node: waiting %.1f seconds", cfg.seconds)
            await asyncio.sleep(cfg.seconds)
            return {}

        return delay_node
