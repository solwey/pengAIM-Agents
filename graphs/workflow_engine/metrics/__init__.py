"""NetSuite SuiteQL metric aggregators used by the calculate_netsuite_metric node.

Each module in this package exposes a single async function with the signature:
    async fn(*, access_token: str, account_id: str, from_date: str, to_date: str) -> dict
"""

from graphs.workflow_engine.metrics.actual_by_department import actual_by_department_metric
from graphs.workflow_engine.metrics.cost_per_sqft import cost_per_sqft_metric

__all__ = [
    "actual_by_department_metric",
    "cost_per_sqft_metric",
]
