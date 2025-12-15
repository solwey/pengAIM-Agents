"""MCP (Model Context Protocol) tools integration for the react_agent.

This module handles loading tools from MCP servers using langchain-mcp-adapters.
Authorization headers are passed to MCP servers using the same pattern as the RAG tool.
"""

from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from graphs.react_agent.context import McpServerConfig


async def load_mcp_tools(
    mcp_servers: list[McpServerConfig],
    authorization: str | None = None,
) -> list[BaseTool]:
    """Load tools from configured MCP servers.

    Args:
        mcp_servers: List of MCP server configurations
        authorization: Authorization token to pass to MCP servers in headers

    Returns:
        List of LangChain tools loaded from all MCP servers
    """
    if not mcp_servers:
        return []

    # Build server configuration for MultiServerMCPClient
    server_config: dict[str, dict[str, Any]] = {}

    for server in mcp_servers:
        config: dict[str, Any] = {
            "transport": "http",
            "url": server.url,
        }

        # Add authorization header if provided
        if authorization:
            config["headers"] = {
                "Authorization": authorization,
            }

        server_config[server.name] = config

    # Load tools from all configured servers
    tools: list[BaseTool] = []

    try:
        client = MultiServerMCPClient(server_config)
        mcp_tools = await client.get_tools()
        tools.extend(mcp_tools)
    except Exception as e:
        print(e)
        # Log error but don't fail the entire tool loading process
        print(f"Warning: Failed to load MCP tools: {e}")

    return tools
