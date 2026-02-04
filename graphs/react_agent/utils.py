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


async def get_api_key_for_model(model_name: str, config: RunnableConfig) -> str | None:
    """Get the appropriate API key based on the model provider.

    Args:
        model_name: Model name in format "provider:model" (e.g., "openai:gpt-4o")
        config: The runnable config containing key information

    Returns:
        The decrypted API key or None if not found
    """
    model_name_lower = model_name.lower()
    model_prefixes = ["openai", "anthropic", "google"]
    provider = next(
        (prefix for prefix in model_prefixes if model_name_lower.startswith(prefix)), None
    )
    if not provider:
        return None

    # Select the appropriate API key based on provider
    if provider == "google":
        key_data = config.get("configurable", {}).get("agent_google_api_key", {})
    else:
        key_data = config.get("configurable", {}).get("agent_openai_api_key", {})

    if not key_data:
        return None

    key = await get_api_key(config, key_data.get("keyId"))
    return key


async def get_api_key(
    config: RunnableConfig,
    key_id: str,
    provider: str | None = None,
    name: str | None = None,
) -> str | None:
    """Get API key from environment or config by key name.

    Args:
        config: The runnable config containing auth information
        key_id: The ID of the key to reveal
        provider: Optional provider name for query params
        name: Optional name for query params

    Returns:
        The decrypted API key or None if not found
    """
    authorization = (
        config.get("configurable", {})
        .get("langgraph_auth_user", {})
        .get("permissions")[0]
        .replace("authz:", "")
    )
    if not authorization or not key_id:
        return None

    search_params = f"provider={provider}&name={name}" if provider and name else ""
    search_endpoint = (
        f"{RAG_URL}/keys/{key_id}/reveal{f'?{search_params}' if search_params else ''}"
    )
    headers = {"authorization": authorization, "Accept": "text/plain"}

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(search_endpoint, headers=headers) as search_response,
        ):
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
