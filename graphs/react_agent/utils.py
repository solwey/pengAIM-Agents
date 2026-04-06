import json
import logging
from datetime import datetime
from typing import Annotated

import httpx
from firecrawl import AsyncFirecrawlApp
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from tavily import AsyncTavilyClient

from aegra_api.settings import settings
from graphs.react_agent.context import AgentMode, Context, SearchAPI
from graphs.react_agent.tools import create_rag_tool

logger = logging.getLogger(__name__)


def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.

    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a %b} {now.day}, {now:%Y}"


def get_provider_from_model_name(model_name: str) -> str | None:
    """Extract the normalized provider prefix from a model name string.

    Args:
        model_name: Model name in format "provider:model"
            (e.g., "openai:gpt-4o", "google_genai:gemini-2.5-pro")

    Returns:
        "openai", "google", "anthropic", or None if unrecognized.
    """
    if not model_name:
        return None
    lower = model_name.lower()
    if lower.startswith("openai:") or lower.startswith("azure_openai:"):
        return "openai"
    if lower.startswith("google_genai:") or lower.startswith("google:"):
        return "google"
    if lower.startswith("anthropic:"):
        return "anthropic"
    return None


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
    provider = next((prefix for prefix in model_prefixes if model_name_lower.startswith(prefix)), None)
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
    authorization = _extract_authorization(config)
    if not authorization or not key_id:
        return None

    search_params = f"provider={provider}&name={name}" if provider and name else ""
    search_endpoint = (
        f"{settings.graphs.RAG_API_URL}/keys/{key_id}/reveal{f'?{search_params}' if search_params else ''}"
    )
    headers = {"authorization": authorization, "Accept": "text/plain"}

    try:
        async with httpx.AsyncClient() as client:
            search_response = await client.get(search_endpoint, headers=headers)
        search_response.raise_for_status()
        key = search_response.text

        return key
    except Exception:
        logger.warning("Failed to fetch API key", exc_info=True)
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
        token = auth_user.get("authorization")
        return token
    except (KeyError, IndexError, AttributeError):
        pass

    return None


def _create_tavily_search_tool():
    """Create a Tavily web search tool for the React agent."""

    @tool(
        name_or_callable="web_search",
        description=(
            "Search the web for current information using Tavily. "
            "Useful for answering questions about current events, "
            "recent data, or facts that require up-to-date information."
        ),
    )
    async def tavily_search(
        query: Annotated[str, "The search query to find relevant information"],
        config: RunnableConfig = None,
    ) -> str:
        api_key = await get_api_key(config, "key_id", "tavily", "root")
        if not api_key:
            return json.dumps({"error": "Tavily API key not configured"})

        client = AsyncTavilyClient(api_key=api_key)
        results = await client.search(query, max_results=5, include_raw_content=False)

        formatted = "Search results:\n\n"
        for i, result in enumerate(results.get("results", [])):
            formatted += f"--- SOURCE {i + 1}: {result.get('title', 'No title')} ---\n"
            formatted += f"URL: {result.get('url', '')}\n"
            formatted += f"Content: {result.get('content', 'No content')}\n\n"

        return formatted if results.get("results") else "No search results found."

    return tavily_search


def _create_firecrawl_search_tool():
    """Create a FireCrawl web search tool for the React agent."""

    @tool(
        name_or_callable="web_search",
        description=(
            "Search the web for current information using FireCrawl. "
            "Useful for answering questions about current events, "
            "recent data, or facts that require up-to-date information."
        ),
    )
    async def firecrawl_search(
        query: Annotated[str, "The search query to find relevant information"],
        config: RunnableConfig = None,
    ) -> str:
        api_key = await get_api_key(config, "api_key", "firecrawl", "root")
        if not api_key:
            return json.dumps({"error": "FireCrawl API key not configured"})

        app = AsyncFirecrawlApp(api_key=api_key)
        response = await app.search(
            query,
            limit=5,
            scrape_options={
                "formats": ["markdown"],
                "only_main_content": True,
                "remove_base64_images": True,
            },
        )

        formatted = "Search results:\n\n"
        items = getattr(response, "web", []) or []
        if not items:
            return "No search results found."

        for i, item in enumerate(items):
            title = getattr(item, "title", "") or "No title"
            url = getattr(item, "url", "") or ""
            markdown = getattr(item, "markdown", "") or ""
            description = getattr(item, "description", "") or ""
            content = markdown if markdown else description

            if not content:
                content = "No content available"

            max_content_length = 2000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "... [truncated]"

            formatted += f"--- SOURCE {i + 1}: {title} ---\n"
            formatted += f"URL: {url}\n"
            formatted += f"Content: {content}\n\n"

        return formatted

    return firecrawl_search


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
        rag_tool = await create_rag_tool(settings.graphs.RAG_API_URL)
        tools.append(rag_tool)

    if cfg.mode == AgentMode.WEB_SEARCH:
        if cfg.search_api == SearchAPI.TAVILY:
            tools.append(_create_tavily_search_tool())
        elif cfg.search_api == SearchAPI.FIRECRAWL:
            tools.append(_create_firecrawl_search_tool())

    if cfg.mcp_tools is not None:
        tools.extend(cfg.mcp_tools)

    return {t.name: t for t in tools}
