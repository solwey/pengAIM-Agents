import os

import aiohttp
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from graphs.tools_agent.context import AgentMode, Context
from graphs.tools_agent.token import fetch_tokens
from graphs.tools_agent.tools import (
    create_langchain_mcp_tool,
    create_rag_tool,
    wrap_mcp_authenticate_tool,
)

RAG_URL = os.getenv("RAG_API_URL", "")

if not RAG_URL:
    raise RuntimeError("RAG-only mode is enabled but RAG_API_URL is not set.")


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


async def _build_tools(
    cfg: Context, config: RunnableConfig
) -> dict[str, StructuredTool]:
    """Build a mapping of tool name -> tool instance based on the config.

    This preserves the previous behavior:
    * In RAG mode, only the RAG `collection` tool is available.
    * In ONLINE mode, we can expose MCP tools (optionally authenticated).
    """
    tools: list[StructuredTool] = []

    if cfg.mode == AgentMode.RAG:
        rag_tool = await create_rag_tool(RAG_URL)
        tools.append(rag_tool)

    mcp_cfg = cfg.mcp_config
    if cfg.mode != AgentMode.RAG and mcp_cfg and mcp_cfg.url and mcp_cfg.tools:
        if mcp_cfg.auth_required:
            mcp_tokens = await fetch_tokens(config)
        else:
            mcp_tokens = None

        headers = (
            mcp_tokens is not None
            and {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
            or None
        )

        server_url = mcp_cfg.url.rstrip("/") + "/mcp"

        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            tool_names_to_find = set(mcp_cfg.tools)
            fetched_mcp_tools_list: list[StructuredTool] = []
            names_of_tools_added = set()

            async with streamablehttp_client(server_url, headers=headers) as streams:
                read_stream, write_stream, _ = streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    page_cursor = None

                    while True:
                        tool_list_page = await session.list_tools(cursor=page_cursor)

                        if not tool_list_page or not tool_list_page.tools:
                            break

                        for mcp_tool in tool_list_page.tools:
                            if not tool_names_to_find or (
                                mcp_tool.name in tool_names_to_find
                                and mcp_tool.name not in names_of_tools_added
                            ):
                                langchain_tool = create_langchain_mcp_tool(
                                    mcp_tool,
                                    mcp_server_url=server_url,
                                    headers=headers,
                                )
                                fetched_mcp_tools_list.append(
                                    wrap_mcp_authenticate_tool(langchain_tool)
                                )
                                if tool_names_to_find:
                                    names_of_tools_added.add(mcp_tool.name)

                        page_cursor = tool_list_page.nextCursor

                        if not page_cursor:
                            break
                        if tool_names_to_find and len(names_of_tools_added) == len(
                            tool_names_to_find
                        ):
                            break

            tools.extend(fetched_mcp_tools_list)
        except Exception as e:
            print(f"Failed to fetch MCP tools: {e}")

    return {t.name: t for t in tools}
