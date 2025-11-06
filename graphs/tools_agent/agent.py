import os
from enum import Enum

import aiohttp
from langchain_core.runnables import RunnableConfig
from typing import Optional, List
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
from langchain_core.tools import StructuredTool

from graphs.tools_agent.token import fetch_tokens
from graphs.tools_agent.tools import wrap_mcp_authenticate_tool, create_langchain_mcp_tool, create_rag_tool

UNEDITABLE_SYSTEM_PROMPT = "\nIf the tool throws an error requiring authentication, provide the user with a Markdown link to the authentication page and prompt them to authenticate."

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that has access to a variety of tools."
)

RAG_URL = os.getenv("RAG_API_URL", "")

if not RAG_URL:
    raise RuntimeError("RAG-only mode is enabled but RAG_API_URL is not set.")


class AgentMode(Enum):
    RAG = "rag"
    ONLINE = "online"

class RetrievalMode(Enum):
    BASIC = "basic"
    HYDE = "hyde"
    RRF = "rrf"


# noinspection PyArgumentList
class MCPConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""


# noinspection PyArgumentList
class GraphConfigPydantic(BaseModel):
    model_name: Optional[str] = Field(
        default="openai:gpt-4o-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "openai:gpt-4o-mini",
                "description": "The model to use in all generations",
                "options": [
                    {
                        "label": "Claude Sonnet 4",
                        "value": "anthropic:claude-sonnet-4-0",
                    },
                    {
                        "label": "Claude 3.7 Sonnet",
                        "value": "anthropic:claude-3-7-sonnet-latest",
                    },
                    {
                        "label": "Claude 3.5 Sonnet",
                        "value": "anthropic:claude-3-5-sonnet-latest",
                    },
                    {
                        "label": "Claude 3.5 Haiku",
                        "value": "anthropic:claude-3-5-haiku-latest",
                    },
                    {"label": "o4 mini", "value": "openai:o4-mini"},
                    {"label": "o3", "value": "openai:o3"},
                    {"label": "o3 mini", "value": "openai:o3-mini"},
                    {"label": "GPT 4o", "value": "openai:gpt-4o"},
                    {"label": "GPT 4o mini", "value": "openai:gpt-4o-mini"},
                    {"label": "GPT 4.1", "value": "openai:gpt-4.1"},
                    {"label": "GPT 4.1 mini", "value": "openai:gpt-4.1-mini"},
                ],
            }
        },
    )
    temperature: Optional[float] = Field(
        default=0.7,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 0.7,
                "min": 0,
                "max": 2,
                "step": 0.1,
                "description": "Controls randomness (0 = deterministic, 2 = creative)",
            }
        },
    )
    max_tokens: Optional[int] = Field(
        default=4000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 4000,
                "min": 1,
                "description": "The maximum number of tokens to generate",
            }
        },
    )
    system_prompt: Optional[str] = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        metadata={
            "x_oap_ui_config": {
                "type": "textarea",
                "placeholder": "Enter a system prompt...",
                "description": f"The system prompt to use in all generations. The following prompt will always be included at the end of the system prompt:\n---{UNEDITABLE_SYSTEM_PROMPT}\n---",
                "default": DEFAULT_SYSTEM_PROMPT,
            }
        },
    )
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                # Here is where you would set the default tools.
                "default": {
                    "tools": [],
                    "url": "http://localhost:4444"
                }
            }
        },
    )
    mode: AgentMode = Field(
        default=AgentMode.RAG,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": AgentMode.RAG.value,
                "description": "Select how the agent retrieves information: from local RAG data or online sources.",
                "options": [
                    {"label": "Rag only", "value": AgentMode.RAG.value},
                    {"label": "Online only", "value": AgentMode.ONLINE.value},
                ]
            }
        },
    )
    rag_system_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "textarea",
                "placeholder": "Enter a custom system prompt or leave empty to use the default one...",
                "description": (
                    "Define a custom system prompt to guide the RAG agent’s behavior and tone. "
                    "If left empty, the agent will automatically use the platform’s default prompt. "
                    "Use this to personalize responses for your specific project or domain."
                ),
                "default": "",
            }
        },
    )
    rag_retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.RRF,
        description="How the agent retrieves information during RAG operations.",
        json_schema_extra={
            "x_oap_ui_config": {
                "type": "select",
                "default": RetrievalMode.RRF.value,
                "description": "Select retrieval strategy for RAG.",
                "options": [
                    {"label": "Basic", "value": RetrievalMode.BASIC.value},
                    {"label": "HyDE", "value": RetrievalMode.HYDE.value},
                    {"label": "RRF", "value": RetrievalMode.RRF.value},
                ],
            }
        },
    )
    agent_openai_api_key: Optional[dict[str, str]] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "password",
                "placeholder": "Enter your custom OpenAI API key for this agent...",
                "description": (
                    "Provide a dedicated OpenAI API key to be used only by this agent. "
                ),
                "default": "",
            }
        },
    )

    rag_openai_api_key: Optional[dict[str, str]] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "password",
                "placeholder": "Enter your OpenAI API key for RAG operations...",
                "description": (
                    "Specify a separate OpenAI API key to be used for RAG tasks "
                    "such as document search, summarization, or contextual QA. "
                ),
                "default": "",
            }
        },
    )


async def get_api_key_for_model(model_name: str, config: RunnableConfig):
    model_name = model_name.lower()
    model_prefixes = [ "openai", "anthropic", "google"]
    key_name = next((prefix for prefix in model_prefixes if model_name.startswith(prefix)), None)
    if not key_name:
        return None

    authorization = config.get("metadata", {}).get("authorization")
    key_data = config.get("configurable", {}).get("agent_openai_api_key", {})

    search_endpoint = f"{RAG_URL}/keys/{key_data.get('keyId')}/reveal"
    headers = {"authorization": authorization, "Accept": "text/plain"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    search_endpoint,
                    headers=headers
            ) as search_response:
                search_response.raise_for_status()
                key = await search_response.text()

        return key
    except Exception as e:
        print("Key exception", e)
        return None


async def graph(config: RunnableConfig):
    cfg = GraphConfigPydantic(**config.get("configurable", {}))
    tools = []

    if cfg.mode == AgentMode.RAG:
        rag_tool = await create_rag_tool(RAG_URL)
        tools.append(rag_tool)

    if cfg.mcp_config and cfg.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    if (
            not cfg.mode == AgentMode.RAG
            and cfg.mcp_config
            and cfg.mcp_config.url
            and cfg.mcp_config.tools
            and (mcp_tokens or not cfg.mcp_config.auth_required)
    ):
        server_url = cfg.mcp_config.url.rstrip("/") + "/mcp"

        tool_names_to_find = set(cfg.mcp_config.tools)
        fetched_mcp_tools_list: list[StructuredTool] = []
        names_of_tools_added = set()

        # If the tokens are not None, then we need to add the authorization header. otherwise make header's None
        headers = (
                mcp_tokens is not None
                and {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
                or None
        )
        try:
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
                                    mcp_tool, mcp_server_url=server_url, headers=headers
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
            pass

    api_key = await get_api_key_for_model(cfg.model_name, config)

    model = init_chat_model(
        cfg.model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        api_key=api_key or "No token found"
    )

    rag_only_contract = ""
    if cfg.mode == AgentMode.RAG:
        rag_only_contract = (
            "\n\n[RAG-ONLY MODE]\n"
            "You must use only the `collection` tool to retrieve context and answer.\n"
            "Do not call or reference any other tools (including MCP tools), web search, or code execution.\n"
            "If the `collection` tool returns no relevant results, reply: \"I don't have enough information in the collection.\" and ask the user to add documents or refine the query.\n"
            "Do not fabricate facts beyond what the tool returns. Prefer quoting short snippets from `documents` when present.\n"
        )
    final_system_prompt = (cfg.system_prompt + rag_only_contract + UNEDITABLE_SYSTEM_PROMPT)

    return create_react_agent(
        prompt=final_system_prompt,
        model=model,
        tools=tools,
        config_schema=GraphConfigPydantic,
    )
