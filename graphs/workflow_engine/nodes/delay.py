"""Delay node executor — pauses workflow execution for a configured duration."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_templates
from graphs.workflow_engine.schema import DelayConfig

logger = logging.getLogger(__name__)


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class DelayExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = DelayConfig(**config)

        async def delay_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})

            sleep_for = float(cfg.seconds)
            if cfg.until_iso:
                resolved = resolve_templates(cfg.until_iso, data)
                target = _parse_iso(resolved)
                if target is None:
                    logger.warning("delay: invalid until_iso '%s', falling back to seconds=%.1f", resolved, cfg.seconds)
                else:
                    if target.tzinfo is None:
                        target = target.replace(tzinfo=UTC)
                    delta = (target - datetime.now(UTC)).total_seconds()
                    sleep_for = max(0.0, min(delta, cfg.max_seconds))
                    logger.info("delay: until %s -> sleeping %.1fs", resolved, sleep_for)

            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            return {}

        return delay_node
