import os
from datetime import datetime

import aiohttp
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from graphs.react_agent.context import AgentMode, Context
from graphs.react_agent.mcp_tools import load_mcp_tools
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


def _extract_authorization(config: RunnableConfig | None) -> str | None:
    """Extract authorization token from RunnableConfig.

    Args:
        config: The RunnableConfig containing auth user context

    Returns:
        Authorization token string or None if not found
    """
    if not config:
        return None

    try:
        configurable = config.get("configurable", {})
        auth_user = configurable.get("langgraph_auth_user", {})
        permissions = auth_user.get("permissions", [])
        if permissions:
            return permissions[0].replace("authz:", "")
    except (KeyError, IndexError, AttributeError):
        pass

    return None


async def _build_tools(
    cfg: Context,
    config: RunnableConfig | None = None,
) -> dict[str, BaseTool]:
    """Build a mapping of tool name -> tool instance based on the config.

    This preserves the previous behavior:
    * In RAG mode, only the RAG `rag_search` tool is available.
    * In ONLINE mode, only simple agent is available.

    Additionally, MCP tools are loaded if configured.

    Args:
        cfg: The Context configuration
        config: Optional RunnableConfig for extracting authorization

    Returns:
        Dictionary mapping tool names to tool instances
    """
    tools: list[BaseTool] = []

    # Load RAG tool if in RAG mode
    if cfg.mode == AgentMode.RAG:
        rag_tool = await create_rag_tool(RAG_URL)
        tools.append(rag_tool)

    # Load MCP tools if configured
    if cfg.mcp_servers:
        authorization = _extract_authorization(config)
        mcp_tools = await load_mcp_tools(cfg.mcp_servers, authorization)
        tools.extend(mcp_tools)

    return {t.name: t for t in tools}
