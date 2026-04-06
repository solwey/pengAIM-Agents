"""Generate Report node executor — triggers presentation/report generation via pengAIM-RAG."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_templates
from graphs.workflow_engine.schema import GenerateReportConfig

logger = logging.getLogger(__name__)


class GenerateReportExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = GenerateReportConfig(**config)

        async def generate_report_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})

            resolved_period = resolve_templates(cfg.period, data) if cfg.period else ""
            resolved_content = resolve_templates(cfg.content, data) if cfg.content else ""

            payload: dict[str, Any] = {
                "report_type": cfg.report_type,
                "period": resolved_period,
                "generator": cfg.generator,
            }
            if cfg.automation_id:
                payload["automation_id"] = cfg.automation_id
            if resolved_content:
                payload["content"] = resolved_content

            # Extract auth token from config if available
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
                    response = await client.post(
                        f"{settings.graphs.RAG_API_URL}/api/v1/reports/generate",
                        json=payload,
                        headers=headers,
                    )

                    if response.status_code in (200, 201):
                        body = response.json()
                        result = {
                            "ok": True,
                            "report_id": body.get("id"),
                            "status": body.get("status"),
                        }
                        logger.info(
                            "Report generation started: report_id=%s",
                            body.get("id"),
                        )
                    else:
                        result = {
                            "ok": False,
                            "status_code": response.status_code,
                            "error": response.text[:500],
                        }
                        logger.warning(
                            "Report generation failed: %d %s",
                            response.status_code,
                            response.text[:200],
                        )

            except httpx.TimeoutException:
                result = {"ok": False, "error": "Report generation request timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return generate_report_node
