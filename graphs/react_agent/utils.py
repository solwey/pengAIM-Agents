import os
from datetime import datetime

import aiohttp
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from graphs.react_agent.context import AgentMode, Context
from graphs.react_agent.tools import create_rag_tool

RAG_URL = os.getenv("RAG_API_URL", "")

if not RAG_URL:
    raise RuntimeError("RAG-only mode is enabled but RAG_API_URL is not set.")


def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.

    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a %b} {now.day}, {now:%Y}"


async def get_api_key_for_model(config: RunnableConfig) -> str | None:
    authorization = (
        config.get("configurable", {})
        .get("langgraph_auth_user", {})
        .get("permissions")[0]
        .replace("authz:", "")
    )
    key_data = config.get("configurable", {}).get("agent_openai_api_key", {})

    if not authorization or not key_data:
        return None

    search_endpoint = f"{RAG_URL}/keys/{key_data.get('keyId')}/reveal"
    headers = {"authorization": authorization, "Accept": "text/plain"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(search_endpoint, headers=headers) as search_response:
                search_response.raise_for_status()
                key = await search_response.text()
        return key
    except Exception as e:
        print("Key exception", e)
        return None


async def _build_tools(cfg: Context) -> dict[str, StructuredTool]:
    """Build a mapping of tool name -> tool instance based on the config.

    This preserves the previous behavior:
    * In RAG mode, only the RAG `rag_search` tool is available.
    * In ONLINE mode, only simple agent is available.
    """
    tools: list[StructuredTool] = []

    if cfg.mode == AgentMode.RAG:
        rag_tool = await create_rag_tool(RAG_URL)
        tools.append(rag_tool)

    return {t.name: t for t in tools}
