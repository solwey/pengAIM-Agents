"""Run Agent node executor — triggers a LangGraph agent run via /runs/wait."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_templates
from graphs.workflow_engine.schema import RunAgentConfig

logger = logging.getLogger(__name__)


class RunAgentExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = RunAgentConfig(**config)

        async def run_agent_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            resolved_prompt = resolve_templates(cfg.prompt, data)

            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")
            base_url = configurable.get("agents_api_url", "http://localhost:8001")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = auth_token

            result: dict[str, Any]
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(cfg.timeout_seconds)
                ) as client:
                    # 1. Create thread
                    thread_resp = await client.post(
                        f"{base_url}/threads",
                        json={},
                        headers=headers,
                    )
                    if thread_resp.status_code not in (200, 201):
                        result = {
                            "ok": False,
                            "error": f"Failed to create thread: {thread_resp.text[:300]}",
                        }
                        return {"data": {**data, cfg.response_key: result}}

                    thread_id = thread_resp.json().get("thread_id")

                    # 2. Create run and wait for completion via /runs/wait
                    wait_resp = await client.post(
                        f"{base_url}/threads/{thread_id}/runs/wait",
                        json={
                            "assistant_id": cfg.assistant_id,
                            "input": {
                                "messages": [
                                    {
                                        "type": "human",
                                        "content": [
                                            {"type": "text", "text": resolved_prompt}
                                        ],
                                    }
                                ]
                            },
                        },
                        headers=headers,
                    )

                    if wait_resp.status_code not in (200, 201):
                        result = {
                            "ok": False,
                            "error": f"Agent run failed: {wait_resp.text[:300]}",
                        }
                        return {"data": {**data, cfg.response_key: result}}

                    output = wait_resp.json()

                    # Extract final message from output
                    agent_output = None
                    messages = output.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        if isinstance(last_msg, dict):
                            agent_output = last_msg.get("content", "")
                        else:
                            agent_output = str(last_msg)

                    result = {
                        "ok": True,
                        "status": "success",
                        "thread_id": thread_id,
                        "output": agent_output,
                    }
                    logger.info("Agent run completed: thread_id=%s", thread_id)

            except httpx.TimeoutException:
                result = {"ok": False, "error": "Agent run timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return run_agent_node
