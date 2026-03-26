"""Run Agent node executor — triggers a LangGraph agent run and waits for completion."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from graphs.workflow_engine.nodes.base import NodeExecutor, resolve_templates
from graphs.workflow_engine.schema import RunAgentConfig

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 3  # seconds between status checks


class RunAgentExecutor(NodeExecutor):
    @staticmethod
    def create(config: dict[str, Any]):
        cfg = RunAgentConfig(**config)

        async def run_agent_node(state: dict, config: RunnableConfig) -> dict:
            data = state.get("data", {})
            resolved_prompt = resolve_templates(cfg.prompt, data)

            # Use internal API (same server) — relative to localhost
            configurable = config.get("configurable", {})
            auth_token = configurable.get("auth_token", "")
            base_url = configurable.get("agents_api_url", "http://localhost:8001")

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"

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

                    # 2. Create run
                    run_resp = await client.post(
                        f"{base_url}/threads/{thread_id}/runs",
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
                    if run_resp.status_code not in (200, 201):
                        result = {
                            "ok": False,
                            "error": f"Failed to create run: {run_resp.text[:300]}",
                        }
                        return {"data": {**data, cfg.response_key: result}}

                    run_data = run_resp.json()
                    run_id = run_data.get("run_id")

                    # 3. Poll for completion
                    elapsed = 0
                    final_status = "unknown"
                    while elapsed < cfg.timeout_seconds:
                        await asyncio.sleep(_POLL_INTERVAL)
                        elapsed += _POLL_INTERVAL

                        status_resp = await client.get(
                            f"{base_url}/threads/{thread_id}/runs/{run_id}",
                            headers=headers,
                        )
                        if status_resp.status_code != 200:
                            continue

                        status_data = status_resp.json()
                        final_status = status_data.get("status", "unknown")

                        if final_status in ("success", "error", "cancelled"):
                            break

                    # 4. Get final messages
                    agent_output = None
                    if final_status == "success":
                        msgs_resp = await client.get(
                            f"{base_url}/threads/{thread_id}/state",
                            headers=headers,
                        )
                        if msgs_resp.status_code == 200:
                            thread_state = msgs_resp.json()
                            values = thread_state.get("values", {})
                            messages = values.get("messages", [])
                            if messages:
                                last_msg = messages[-1]
                                if isinstance(last_msg, dict):
                                    agent_output = last_msg.get("content", "")
                                else:
                                    agent_output = str(last_msg)

                    result = {
                        "ok": final_status == "success",
                        "status": final_status,
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "output": agent_output,
                    }

                    if final_status == "success":
                        logger.info("Agent run completed: run_id=%s", run_id)
                    else:
                        logger.warning("Agent run ended with status=%s, run_id=%s", final_status, run_id)

            except httpx.TimeoutException:
                result = {"ok": False, "error": "Agent run timed out"}
            except httpx.RequestError as exc:
                result = {"ok": False, "error": f"Request failed: {exc}"}

            return {"data": {**data, cfg.response_key: result}}

        return run_agent_node
