"""MCP (Model Context Protocol) shared infrastructure.

Provides:
- McpConfigMixin: Pydantic mixin for graphs that support MCP tools.
  Any graph whose config_schema includes this mixin will have MCP tools
  pre-loaded with persistent sessions before graph execution.

- build_mcp_client / load_tools_with_sessions: helpers used by the
  server to set up sessions and load tools.

- load_mcp_tools: sessionless fallback for standalone usage.
"""

import logging
from contextlib import AsyncExitStack
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools as _adapter_load_mcp_tools
from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# MCP server config model
# ------------------------------------------------------------------

class McpServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    name: str = Field(..., description="Unique name for this MCP server")
    url: str = Field(..., description="HTTP URL of the MCP server endpoint")


# ------------------------------------------------------------------
# Mixin — any graph config_schema that inherits this gets MCP support
# ------------------------------------------------------------------

class McpConfigMixin:
    """Mixin for graph config schemas that support MCP tools.

    Graphs whose config_schema includes this mixin will automatically
    have MCP tools pre-loaded with persistent sessions in execute_run_async.
    """

    mcp_servers: list[McpServerConfig] = Field(
        default_factory=list,
        description="List of MCP servers to load tools from",
        metadata={
            "x_oap_ui_config": {
                "type": "json",
                "description": (
                    "Configure MCP (Model Context Protocol) servers to extend the agent's "
                    "capabilities with additional tools. Each server should expose tools "
                    "via HTTP endpoints."
                ),
                "default": [],
            }
        },
    )

    # Pre-loaded MCP tools injected by execute_run_async with persistent sessions.
    # Not part of the user-facing config — excluded from serialization/schema.
    mcp_tools: list | None = Field(default=None, exclude=True)

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def validate_mcp_servers(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [
                McpServerConfig(**item) if isinstance(item, dict) else item
                for item in v
            ]
        return v


# ------------------------------------------------------------------
# Client helpers
# ------------------------------------------------------------------

def build_mcp_client(
    mcp_servers: list[McpServerConfig],
    authorization: str | None = None,
) -> MultiServerMCPClient:
    """Create a MultiServerMCPClient from server configs."""
    server_config: dict[str, dict[str, Any]] = {}

    for server in mcp_servers:
        config: dict[str, Any] = {
            "transport": "http",
            "url": server.url,
        }
        if authorization:
            config["headers"] = {
                "Authorization": authorization,
            }
        server_config[server.name] = config

    return MultiServerMCPClient(server_config)


async def load_tools_with_sessions(
    client: MultiServerMCPClient,
    server_names: list[str],
    stack: AsyncExitStack,
) -> list[BaseTool]:
    """Open persistent sessions and load tools bound to them.

    Sessions are entered on the provided AsyncExitStack, so they stay
    alive until the stack is closed.
    """
    tools: list[BaseTool] = []

    for server_name in server_names:
        try:
            session = await stack.enter_async_context(client.session(server_name))
            server_tools = await _adapter_load_mcp_tools(
                session, server_name=server_name
            )
            tools.extend(server_tools)
        except Exception as e:
            logger.exception(
                "Failed to load MCP tools from server '%s': %s", server_name, e
            )

    return tools


async def load_mcp_tools(
    mcp_servers: list[McpServerConfig],
    authorization: str | None = None,
) -> list[BaseTool]:
    """Load tools from configured MCP servers (sessionless fallback).

    Creates ephemeral sessions per tool call. Used only when tools
    aren't pre-loaded via the mixin's mcp_tools field.
    """
    if not mcp_servers:
        return []

    client = build_mcp_client(mcp_servers, authorization)
    tools: list[BaseTool] = []

    try:
        mcp_tools = await client.get_tools()
        tools.extend(mcp_tools)
    except Exception as e:
        logger.exception("Failed to load MCP tools: %s", e)

    return tools
