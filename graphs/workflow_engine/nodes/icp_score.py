"""ICP Score node — scores an account against the team's ICP using LLM."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field
from graphs.workflow_engine.schema import ICPScoreConfig

logger = logging.getLogger(__name__)

ICP_SCORING_PROMPT = """You are an ICP (Ideal Customer Profile) scoring assistant.

Given the company's ICP definition and an account's data, score the account from 0 to 100 on how well it matches the ICP.

## Team's ICP:
{icp}

## Buyer Personas:
{personas}

## Account Data:
{account_data}

Respond with ONLY a JSON object (no markdown, no explanation):
{{"score": <0-100>, "status": "<hot|warm|cold>", "reasoning": "<1-2 sentence explanation>"}}

Rules:
- score >= 80 → status "hot"
- score 50-79 → status "warm"
- score < 50 → status "cold"
"""


class ICPScoreExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = ICPScoreConfig(**config)

        async def icp_score_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            # 1. Get account data from state
            if cfg.account_data_key:
                account_data = resolve_field(data, cfg.account_data_key)
            else:
                account_data = data

            if not account_data:
                return {"data": {**data, cfg.response_key: {"ok": False, "error": "No account data found"}}}

            # 2. Fetch team context (ICP + buyer personas) from backend
            icp = ""
            personas = ""
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
                    resp = await client.get(
                        f"{settings.graphs.REVY_API_URL}/api/v1/team-context",
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        ctx = resp.json()
                        icp = ctx.get("ideal_customer_profile") or "Not defined"
                        personas = ctx.get("buyer_personas") or "Not defined"
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                logger.warning("Failed to fetch team context: %s", exc)
                icp = "Not available"
                personas = "Not available"

            # 3. Call LLM via init_chat_model
            prompt = ICP_SCORING_PROMPT.format(
                icp=icp,
                personas=personas,
                account_data=json.dumps(account_data, indent=2, default=str)[:3000],
            )

            result: dict[str, Any]
            try:
                llm_model = cfg.model or settings.graphs.WORKFLOW_LLM_MODEL
                model = init_chat_model(
                    llm_model,
                    temperature=0,
                    max_tokens=200,
                )
                response = await model.ainvoke([HumanMessage(content=prompt)])
                content = str(response.content or "")

                # Parse LLM JSON response
                try:
                    clean = content.strip()
                    if clean.startswith("```"):
                        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
                    parsed = json.loads(clean)
                    result = {
                        "ok": True,
                        "score": int(parsed.get("score", 0)),
                        "status": parsed.get("status", "cold"),
                        "reasoning": parsed.get("reasoning", ""),
                    }
                    logger.info("ICP score: %d (%s)", result["score"], result["status"])
                except (json.JSONDecodeError, ValueError) as exc:
                    result = {
                        "ok": False,
                        "error": f"Failed to parse LLM response: {exc}",
                        "raw_response": content[:500],
                    }

            except Exception as exc:  # LLM providers raise diverse exceptions
                logger.warning("ICP score LLM call failed: %s", exc)
                result = {"ok": False, "error": str(exc)[:500]}

            return {"data": {**data, cfg.response_key: result}}

        return icp_score_node
