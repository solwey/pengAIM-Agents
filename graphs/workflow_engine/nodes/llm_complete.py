"""LLM Complete node — generic LLM call via langchain init_chat_model."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_templates
from graphs.workflow_engine.schema import LLMCompleteConfig

logger = logging.getLogger(__name__)


_ROLE_TO_MESSAGE: dict[str, type[BaseMessage]] = {
    "system": SystemMessage,
    "user": HumanMessage,
    "human": HumanMessage,
    "assistant": AIMessage,
    "ai": AIMessage,
}


def _build_messages(cfg: LLMCompleteConfig, data: dict[str, Any]) -> list[BaseMessage]:
    if cfg.messages:
        out: list[BaseMessage] = []
        for raw in cfg.messages:
            role = (raw.get("role") or "user").lower()
            content = resolve_templates(raw.get("content", "") or "", data)
            cls = _ROLE_TO_MESSAGE.get(role)
            if cls is None:
                raise ValueError(f"Unknown message role '{role}'; expected one of {sorted(_ROLE_TO_MESSAGE)}")
            out.append(cls(content=content))
        return out

    out2: list[BaseMessage] = []
    if cfg.system_prompt:
        out2.append(SystemMessage(content=resolve_templates(cfg.system_prompt, data)))
    out2.append(HumanMessage(content=resolve_templates(cfg.prompt, data)))
    return out2


class LLMCompleteExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = LLMCompleteConfig(**config)

        async def llm_complete_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})

            try:
                messages = _build_messages(cfg, data)
            except ValueError as exc:
                return {"data": {**data, cfg.response_key: {"ok": False, "error": str(exc)}}}

            llm_model = cfg.model or settings.graphs.WORKFLOW_LLM_MODEL
            init_kwargs: dict[str, Any] = {
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
            }
            if cfg.response_format == "json":
                init_kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

            result: dict[str, Any]
            try:
                model = init_chat_model(llm_model, **init_kwargs)
                response = await asyncio.wait_for(model.ainvoke(messages), timeout=cfg.timeout_seconds)
                content = str(response.content or "")

                usage: dict[str, int] = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = {
                        "input_tokens": response.usage_metadata.get("input_tokens", 0),
                        "output_tokens": response.usage_metadata.get("output_tokens", 0),
                    }

                result = {"ok": True, "content": content, "usage": usage}
                logger.info("LLM complete: %d chars response", len(content))

            except TimeoutError:
                logger.warning(
                    "LLM complete timed out after %ds (model=%s)",
                    cfg.timeout_seconds,
                    llm_model,
                )
                result = {"ok": False, "error": f"LLM call timed out after {cfg.timeout_seconds}s"}
            except Exception as exc:  # noqa: BLE001  # LLM SDKs raise provider-specific exceptions
                logger.warning("LLM complete failed: %s", exc)
                result = {"ok": False, "error": str(exc)[:500]}

            return {"data": {**data, cfg.response_key: result}}

        return llm_complete_node
