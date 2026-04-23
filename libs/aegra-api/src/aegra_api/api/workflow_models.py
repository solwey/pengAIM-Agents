"""Workflow model health endpoint — probes each supported LLM model and reports reachability."""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, Query
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ..core.auth_deps import auth_dependency, get_current_user
from ..llm_models import LLM_MODELS
from ..models.auth import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Workflow Models"], dependencies=auth_dependency)


MODEL_PROBE_TIMEOUT_SECONDS = 15.0
CACHE_TTL_SECONDS = 300

ModelStatus = Literal["ok", "no_access", "quota_exceeded", "timeout", "error"]


class ModelHealth(BaseModel):
    value: str
    label: str
    status: ModelStatus
    error_message: str | None = None
    latency_ms: int | None = None


class WorkflowModelsHealthResponse(BaseModel):
    models: list[ModelHealth]
    checked_at: str
    cache_ttl_seconds: int = CACHE_TTL_SECONDS


_cache: dict[str, tuple[float, WorkflowModelsHealthResponse]] = {}


def _classify_error(exc: BaseException) -> tuple[ModelStatus, str]:
    err_type = type(exc).__name__
    msg = str(exc)[:300] or err_type
    lowered = f"{err_type} {msg}".lower()

    if isinstance(exc, asyncio.TimeoutError):
        return "timeout", "Model did not respond within timeout"
    if "permissiondenied" in lowered or "403" in lowered or "does not have access" in lowered:
        return "no_access", msg
    if (
        "resource_exhausted" in lowered
        or "429" in lowered
        or "quota" in lowered
        or "rate limit" in lowered
        or "ratelimit" in lowered
    ):
        return "quota_exceeded", msg
    return "error", f"{err_type}: {msg}"


async def _probe_model(value: str, label: str) -> ModelHealth:
    t0 = time.monotonic()
    try:
        model = init_chat_model(value, temperature=0, max_tokens=5)
        await asyncio.wait_for(
            model.ainvoke([HumanMessage(content="hi")]),
            timeout=MODEL_PROBE_TIMEOUT_SECONDS,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        return ModelHealth(value=value, label=label, status="ok", latency_ms=latency_ms)
    except (asyncio.CancelledError, KeyboardInterrupt):
        raise
    except BaseException as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        status, error_message = _classify_error(exc)
        logger.info("Model probe %s -> %s (%s)", value, status, error_message[:120])
        return ModelHealth(
            value=value,
            label=label,
            status=status,
            error_message=error_message,
            latency_ms=latency_ms,
        )


@router.get("/workflow-models/health", response_model=WorkflowModelsHealthResponse)
async def get_workflow_models_health(
    force_refresh: bool = Query(default=False),
    user: User = Depends(get_current_user),
) -> WorkflowModelsHealthResponse:
    """Probe each supported LLM model and report reachability.

    Results are cached server-wide for 5 minutes (env-level API keys are shared).
    Pass ?force_refresh=true to bypass the cache.
    """
    cache_key = "global"
    now = time.monotonic()

    if not force_refresh:
        cached = _cache.get(cache_key)
        if cached is not None and cached[0] > now:
            return cached[1]

    probes = [_probe_model(value, label) for value, label in LLM_MODELS]
    results = await asyncio.gather(*probes)

    response = WorkflowModelsHealthResponse(
        models=results,
        checked_at=datetime.now(UTC).isoformat(),
    )
    _cache[cache_key] = (now + CACHE_TTL_SECONDS, response)
    return response
