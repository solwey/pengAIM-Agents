"""LLM Complete node — generic LLM call via langchain init_chat_model."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_templates
from graphs.workflow_engine.schema import LLMCompleteConfig

logger = logging.getLogger(__name__)

# Model name in "provider:model" format (e.g. "openai:gpt-4o-mini")
WORKFLOW_LLM_MODEL = os.getenv("WORKFLOW_LLM_MODEL", "openai:gpt-4o-mini")


class LLMCompleteExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = LLMCompleteConfig(**config)

        async def llm_complete_node(
            state: dict, config: RunnableConfig
        ) -> dict:
            data = state.get("data", {})

            # Resolve template variables in prompt and system_prompt
            prompt = resolve_templates(cfg.prompt, data)
            system_prompt = (
                resolve_templates(cfg.system_prompt, data)
                if cfg.system_prompt
                else ""
            )

            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            result: dict[str, Any]
            try:
                model = init_chat_model(
                    WORKFLOW_LLM_MODEL,
                    temperature=0,
                    max_tokens=cfg.max_tokens,
                )
                response = await model.ainvoke(messages)
                content = response.content or ""

                usage = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage = {
                        "input_tokens": response.usage_metadata.get(
                            "input_tokens", 0
                        ),
                        "output_tokens": response.usage_metadata.get(
                            "output_tokens", 0
                        ),
                    }

                result = {
                    "ok": True,
                    "content": content,
                    "usage": usage,
                }
                logger.info(
                    "LLM complete: %d chars response", len(content)
                )

            except Exception as exc:
                logger.warning("LLM complete failed: %s", exc)
                result = {"ok": False, "error": str(exc)[:500]}

            return {"data": {**data, cfg.response_key: result}}

        return llm_complete_node