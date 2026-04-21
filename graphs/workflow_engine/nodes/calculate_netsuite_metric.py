"""Calculate NetSuite Metric node — stub.

Placeholder executor: logs and writes a pending-status payload to state.
Real metric aggregation will be implemented in a follow-up.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor
from graphs.workflow_engine.schema import CalculateNetsuiteMetricConfig


class CalculateNetsuiteMetricExecutor(NodeExecutor):
    """Stub — emits a placeholder result. No NetSuite calls yet."""

    @staticmethod
    def create(config: dict[str, Any]):
        cfg = CalculateNetsuiteMetricConfig(**config)

        async def calculate_netsuite_metric_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
            data: dict[str, Any] = state.get("data", {})

            print("Calculating")

            return {
                "data": {
                    **data,
                    cfg.response_key: {
                        "ok": True,
                        "status": "pending",
                        "metric": cfg.metric,
                        "period": cfg.period,
                    },
                }
            }

        return calculate_netsuite_metric_node
