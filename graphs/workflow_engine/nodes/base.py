"""Base class for all workflow node executors.

Every new node type should subclass NodeExecutor and implement `create()`.
This makes adding new node types trivial — just create a new file,
subclass NodeExecutor, and register in nodes/__init__.py.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings

logger = logging.getLogger(__name__)


# Default retry-able status codes — covers Instantly/RAG transient failures
DEFAULT_RETRY_STATUS: tuple[int, ...] = (429, 500, 502, 503, 504)


async def http_request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json: Any | None = None,
    params: dict[str, Any] | None = None,
    timeout_seconds: int = 30,
    max_attempts: int = 3,
    retry_on_status: tuple[int, ...] = DEFAULT_RETRY_STATUS,
    op_name: str = "request",
) -> httpx.Response:
    """Issue an HTTP request with exponential backoff on 429/5xx + timeout.

    Returns the final response (which the caller still needs to validate via
    ``raise_for_status``). Honors the ``Retry-After`` header for 429s.

    Raises the last httpx exception if all attempts fail.
    """
    last_exc: BaseException | None = None
    last_resp: httpx.Response | None = None
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
                resp = await client.request(
                    method.upper(),
                    url,
                    headers=headers,
                    json=json,
                    params=params,
                )
            last_resp = resp
            if resp.status_code not in retry_on_status:
                return resp

            # Compute backoff: prefer Retry-After when available
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = 2.0 * (2**attempt)
            else:
                delay = 2.0 * (2**attempt)
            # ±25% jitter to avoid thundering herd (not security-sensitive)
            delay = delay * (1.0 + random.uniform(-0.25, 0.25))  # noqa: S311  # nosec B311
            logger.warning(
                "%s: %s %s -> %d, retry %d/%d in %.2fs",
                op_name,
                method,
                url,
                resp.status_code,
                attempt + 1,
                max_attempts,
                delay,
            )
            if attempt + 1 < max_attempts:
                await asyncio.sleep(delay)
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            last_exc = exc
            logger.warning(
                "%s: %s %s network error (%s), retry %d/%d",
                op_name,
                method,
                url,
                exc,
                attempt + 1,
                max_attempts,
            )
            if attempt + 1 < max_attempts:
                await asyncio.sleep(2.0 * (2**attempt))

    if last_resp is not None:
        return last_resp
    assert last_exc is not None
    raise last_exc


class NodeExecutor(ABC):
    """Abstract base for workflow node executors.

    Subclasses must implement `create(config)` which returns an async
    node function compatible with LangGraph's `StateGraph.add_node()`.
    """

    @staticmethod
    @abstractmethod
    def create(config: dict[str, Any]) -> Callable[..., Coroutine[Any, Any, dict]]:
        """Build and return an async node function from the given config.

        The returned function signature must be:
            async def node_fn(state: dict, config: RunnableConfig) -> dict
        """
        ...


# ── Shared utilities ─────────────────────────────────────────


_TEMPLATE_RE = re.compile(r"\{\{(\w+(?:\.\w+)*)\}\}")


def resolve_field(data: dict[str, Any], field_path: str) -> Any:
    """Resolve a dot-notation path against a nested dict.

    Example:
        resolve_field({"api_response": {"status_code": 200}}, "api_response.status_code")
        → 200
    """
    parts = field_path.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (IndexError, ValueError):
                return None
        else:
            return None
    return current


def resolve_templates(value: str, data: dict[str, Any]) -> str:
    """Replace {{key}} or {{key.subkey}} placeholders with values from data.

    Only resolves from state data — never from environment variables.
    """

    def _replacer(match: re.Match) -> str:
        path = match.group(1)
        resolved = resolve_field(data, path)
        return str(resolved) if resolved is not None else match.group(0)

    return _TEMPLATE_RE.sub(_replacer, value)


# Comparison operators — whitelist only, no eval/exec
OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: a is not None and b is not None and a > b,
    "lt": lambda a, b: a is not None and b is not None and a < b,
    "gte": lambda a, b: a is not None and b is not None and a >= b,
    "lte": lambda a, b: a is not None and b is not None and a <= b,
    "contains": lambda a, b: b in a if a is not None else False,
    "not_contains": lambda a, b: b not in a if a is not None else True,
}


def compare(actual: Any, operator: str, expected: Any) -> bool:
    """Safely compare two values using a whitelisted operator."""
    fn = OPERATORS.get(operator)
    if fn is None:
        raise ValueError(f"Unknown operator: '{operator}'")
    try:
        return fn(actual, expected)
    except TypeError:
        return False


async def reveal_api_key(config: RunnableConfig, key_id: str) -> str | None:
    """Fetch a decrypted API key from pengAIM-RAG via /keys/{key_id}/reveal.

    Uses auth_token from RunnableConfig.configurable to authenticate.
    Returns None if RAG_API_URL is not set, auth is missing, or fetch fails.
    """
    if not key_id:
        return None

    auth_token = config.get("configurable", {}).get("auth_token", "")
    if not auth_token:
        return None

    url = f"{settings.graphs.RAG_API_URL}/keys/{key_id}/reveal"
    headers = {"authorization": auth_token, "Accept": "text/plain"}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.text
    except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.warning("Failed to reveal api key %s: %s", key_id, exc)
        return None


async def fetch_ingestion_configurable(auth_token: str) -> dict[str, Any]:
    """Read the team's Ingestion Configuration and return a configurable-ready dict.

    Populates the same keys that react_agent/open_deep_research read from
    ``config.configurable`` — ``llm_provider``, ``llm_model``, and either
    ``rag_openai_api_key`` or ``rag_google_api_key`` with ``{"keyId": ...}``.
    Returns an empty dict on any failure — callers should merge with their
    existing configurable and fall back gracefully.
    """
    if not auth_token:
        return {}

    url = f"{settings.graphs.RAG_API_URL}/integrations/ingestion_config"
    headers = {"authorization": auth_token, "Accept": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10)) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                return {}
            cfg = (resp.json() or {}).get("config_json") or {}
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        logger.warning("Failed to fetch ingestion config: %s", exc)
        return {}

    provider = "google" if (cfg.get("llm_provider") or "").lower() in ("gemini", "google") else "openai"
    api_key_id = cfg.get("api_key_id") or ""
    model_name = cfg.get("llm_model") or ""

    out: dict[str, Any] = {"llm_provider": provider}
    if api_key_id:
        key_field = "rag_google_api_key" if provider == "google" else "rag_openai_api_key"
        out[key_field] = {"keyId": api_key_id}
    if model_name:
        # LangChain's init_chat_model expects a prefixed string like "openai:gpt-4o-mini"
        prefix = "google_genai" if provider == "google" else "openai"
        out["llm_model"] = f"{prefix}:{model_name}"
    return out
