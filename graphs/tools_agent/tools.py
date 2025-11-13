import json
from typing import Annotated

import aiohttp
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


async def create_rag_tool(rag_url: str):
    """Create a RAG tool that calls a unified QA endpoint.

    Args:
        rag_url: The base URL for the RAG API server

    Returns:
        A structured tool that accepts a user query and returns a JSON string with
        {response: str, reasoning: str}.
    """
    if rag_url.endswith("/"):
        rag_url = rag_url[:-1]

    try:
        collection_description = "Search your collection of documents for results semantically similar to the input query"

        @tool(name_or_callable="rag_search", description=collection_description)
        async def get_documents(
            query: Annotated[str, "The search query to find relevant documents"],
            config: RunnableConfig = None,
        ) -> str:
            """Search for documents in the collection based on the query"""

            authorization = (
                config.get("configurable", {})
                .get("langgraph_auth_user", {})
                .get("permissions")[0]
                .replace("authz:", "")
            )

            system_prompt = (
                config.get("configurable", {}).get("rag_system_prompt") or None
            )
            retrieval_mode = (
                config.get("configurable", {}).get("rag_retrieval_mode") or None
            )
            key_data = config.get("configurable", {}).get("rag_openai_api_key", {})

            search_endpoint = f"{rag_url}/query"
            body = {
                "question": query,
                "system_prompt": system_prompt,
                "retrieval_mode": retrieval_mode,
                "api_key_id": key_data.get("keyId"),
            }
            headers = {"authorization": authorization}

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        search_endpoint, json=body, headers=headers
                    ) as search_response:
                        search_response.raise_for_status()
                        data = await search_response.json()

                out = {
                    "response": data.get("response", "") or "",
                    "reasoning": data.get("reasoning", "") or "",
                    "context": data.get("context", {}) or {},
                }

                return json.dumps(out, ensure_ascii=False)
            except Exception as e:
                return json.dumps(
                    {"error": f"RAG tool call failed: {str(e)}"}, ensure_ascii=False
                )

        return get_documents

    except Exception as e:
        raise Exception(f"Failed to create RAG tool: {str(e)}")
