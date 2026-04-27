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
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_field, reveal_api_key
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
{custom_criteria_section}
Respond with ONLY a JSON object (no markdown, no explanation):
{{"score": <0-100>, "reasoning": "<1-2 sentence explanation>"}}
"""

CUSTOM_CRITERIA_SECTION = """
## Additional Scoring Criteria:
{criteria}
"""


async def _fetch_team_context(auth_token: str) -> tuple[str, str]:
    """Fetch ICP and personas from solwey backend. Returns ("Not available", ...) on failure."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = auth_token

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
                return icp, personas
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        logger.warning("Failed to fetch team context: %s", exc)
        return "Not available", "Not available"
    return "Not defined", "Not defined"


async def resolve_llm_key(config: RunnableConfig) -> str | None:
    """Reveal the LLM API key from ``configurable`` following the react_agent convention.

    Reads ``llm_provider`` + ``rag_openai_api_key.keyId`` / ``rag_google_api_key.keyId``
    (populated from Ingestion Configuration by the workflow trigger). Matches
    ``graphs/open_deep_research/utils.py::rag_search`` exactly.
    """
    configurable = config.get("configurable", {})
    provider = configurable.get("llm_provider", "openai")
    key_field = "rag_google_api_key" if provider == "google" else "rag_openai_api_key"
    key_data = configurable.get(key_field) or {}
    key_id = key_data.get("keyId")
    return await reveal_api_key(config, key_id) if key_id else None


def _classify(score: int, hot_threshold: int, warm_threshold: int) -> str:
    if score >= hot_threshold:
        return "hot"
    if score >= warm_threshold:
        return "warm"
    return "cold"


async def score_account(
    account_data: Any,
    *,
    model: str,
    hot_threshold: int,
    warm_threshold: int,
    custom_criteria: str,
    auth_token: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Score one account against the team ICP. Status is recomputed from thresholds.

    Used by both the icp_score node and the preview endpoint.
    """
    if not account_data:
        return {"ok": False, "error": "No account data provided"}

    icp, personas = await _fetch_team_context(auth_token)

    custom_section = CUSTOM_CRITERIA_SECTION.format(criteria=custom_criteria.strip()) if custom_criteria.strip() else ""
    prompt = ICP_SCORING_PROMPT.format(
        icp=icp,
        personas=personas,
        account_data=json.dumps(account_data, indent=2, default=str)[:3000],
        custom_criteria_section=custom_section,
    )

    try:
        llm_model = model or settings.graphs.WORKFLOW_LLM_MODEL
        init_kwargs: dict[str, Any] = {"temperature": 0, "max_tokens": 200}
        if api_key:
            init_kwargs["api_key"] = api_key
        llm = init_chat_model(llm_model, **init_kwargs)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = str(response.content or "")
    except Exception as exc:  # LLM providers raise diverse exceptions
        logger.warning("ICP score LLM call failed: %s", exc)
        return {"ok": False, "error": str(exc)[:500]}

    try:
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        parsed = json.loads(clean)
        score = int(parsed.get("score", 0))
        score = max(0, min(100, score))
        result = {
            "ok": True,
            "score": score,
            "status": _classify(score, hot_threshold, warm_threshold),
            "reasoning": parsed.get("reasoning", ""),
        }
        logger.info("ICP score: %d (%s)", result["score"], result["status"])
        return result
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "ok": False,
            "error": f"Failed to parse LLM response: {exc}",
            "raw_response": content[:500],
        }


class ICPScoreExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = ICPScoreConfig(**config)

        async def icp_score_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")

            if cfg.account_data_key:
                account_data = resolve_field(data, cfg.account_data_key)
            else:
                account_data = data

            api_key = await resolve_llm_key(config)
            effective_model = cfg.model or configurable.get("llm_model") or ""

            result = await score_account(
                account_data,
                model=effective_model,
                hot_threshold=cfg.hot_threshold,
                warm_threshold=cfg.warm_threshold,
                custom_criteria=cfg.custom_criteria,
                auth_token=auth_token,
                api_key=api_key,
            )
            return {"data": {**data, cfg.response_key: result}}

        return icp_score_node
