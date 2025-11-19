import asyncio
import json
from typing import Annotated

import aiohttp
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import ValidationError

from graphs.react_agent.rag_models import RagToolResponse, RagToolError, SourceDocument, DocumentCollectionInfo


async def create_rag_tool(rag_url: str):
    """Create a RAG tool that calls a unified QA endpoint.

    Args:
        rag_url: The base URL for the RAG API server

    Returns:
        A structured tool that returns RagToolResponse with context and sources,
        or RagToolError if the request fails.
    """
    if rag_url.endswith("/"):
        rag_url = rag_url[:-1]

    try:
        collection_description = (
            "Search your collection of documents for results semantically similar "
            "to the input query. Returns structured context with sources and metadata."
        )

        @tool(name_or_callable="rag_search", description=collection_description)
        async def get_documents(
            query: Annotated[str, "The search query to find relevant documents"],
            config: RunnableConfig = None,
        ) -> str:
            """Search for documents in the collection based on the query.

            Returns:
                JSON-encoded RagToolResponse or RagToolError
            """

            # === STEP 1: Extract and validate configuration ===
            try:
                try:
                    configurable = config.get("configurable", {}) if config else {}

                    # Extract auth token with validation
                    auth_user = configurable.get("langgraph_auth_user", {})
                    permissions = auth_user.get("permissions", [])
                    if not permissions:
                        raise ValueError("No permissions found in auth user context")

                    authorization = permissions[0].replace("authz:", "")

                    # Extract optional parameters
                    system_prompt = configurable.get("rag_system_prompt")
                    retrieval_mode = configurable.get("rag_retrieval_mode")
                    key_data = configurable.get("rag_openai_api_key", {})

                except (KeyError, IndexError, AttributeError) as e:
                    error = RagToolError(
                        error=f"Configuration extraction failed: {str(e)}",
                        error_type="config_error",
                        details={
                            "config_keys": list(configurable.keys())
                            if "configurable" in locals()
                            else []
                        },
                    )
                    return json.dumps(error.model_dump(), ensure_ascii=False)

                # === STEP 2: Prepare API request ===
                search_endpoint = f"{rag_url}/query/retrieve"
                body = {
                    "question": query,
                    "system_prompt": system_prompt,
                    "retrieval_mode": retrieval_mode,
                    "api_key_id": key_data.get("keyId"),
                }
                headers = {
                    "authorization": authorization,
                    "Content-Type": "application/json",
                }

                # === STEP 3: Make HTTP request with proper error handling ===
                try:
                    timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            search_endpoint, json=body, headers=headers
                        ) as response:
                            # Check HTTP status
                            if response.status == 401:
                                error = RagToolError(
                                    error="Authentication failed. Please check your credentials.",
                                    error_type="auth_error",
                                    details={"status_code": 401},
                                )
                                return json.dumps(error.model_dump(), ensure_ascii=False)

                            if response.status == 404:
                                error = RagToolError(
                                    error="RAG endpoint not found. Check RAG_API_URL configuration.",
                                    error_type="endpoint_error",
                                    details={"status_code": 404, "url": search_endpoint},
                                )
                                return json.dumps(error.model_dump(), ensure_ascii=False)

                            response.raise_for_status()

                            # Parse response
                            data = await response.json()

                except asyncio.TimeoutError:
                    error = RagToolError(
                        error="RAG request timed out after 30 seconds",
                        error_type="timeout_error",
                    )
                    return json.dumps(error.model_dump(), ensure_ascii=False)

                except aiohttp.ClientError as e:
                    error = RagToolError(
                        error=f"Network error: {str(e)}",
                        error_type="network_error",
                        details={"exception": type(e).__name__},
                    )
                    return json.dumps(error.model_dump(), ensure_ascii=False)

                except json.JSONDecodeError as e:
                    error = RagToolError(
                        error="Invalid JSON response from RAG API",
                        error_type="parse_error",
                        details={"exception": str(e)},
                    )
                    return json.dumps(error.model_dump(), ensure_ascii=False)

                # === STEP 4: Validate and transform response ===
                try:
                    # Validate response structure
                    if not isinstance(data, dict):
                        raise ValueError(f"Expected dict response, got {type(data)}")

                    # Parse sources if present
                    sources = []
                    raw_sources = data.get("sources", [])
                    if isinstance(raw_sources, list):
                        for src_data in raw_sources:
                            try:
                                source = SourceDocument(**src_data)
                                sources.append(source)
                            except ValidationError as e:
                                # Log but don't fail on individual source errors
                                print(f"Warning: Invalid source document: {e}")
                                continue

                    # Parse document collections if present
                    document_collections = []
                    raw_collections = data.get("document_collections", [])
                    if isinstance(raw_collections, list):
                        for coll_data in raw_collections:
                            try:
                                collection = DocumentCollectionInfo(**coll_data)
                                document_collections.append(collection)
                            except ValidationError as e:
                                # Log but don't fail on individual collection errors
                                print(f"Warning: Invalid document collection: {e}")
                                continue

                    # Build structured response
                    rag_response = RagToolResponse(
                        context_text=data.get("context_text", ""),
                        sources=sources,
                        retrieval_metadata=data.get("retrieval_metadata", {}),
                        document_collections=document_collections,
                    )

                    # Return structured JSON
                    return rag_response.model_dump_json(indent=2)

                except (ValidationError, ValueError) as e:
                    error = RagToolError(
                        error=f"Response validation failed: {str(e)}",
                        error_type="validation_error",
                        details={
                            "raw_data_keys": list(data.keys())
                            if isinstance(data, dict)
                            else []
                        },
                    )
                    return json.dumps(error.model_dump(), ensure_ascii=False)

            except Exception as e:
                # Catch-all for unexpected errors
                error = RagToolError(
                    error=f"Unexpected error: {str(e)}",
                    error_type="unknown_error",
                    details={"exception": type(e).__name__},
                )
                return json.dumps(error.model_dump(), ensure_ascii=False)

        return get_documents

    except Exception as e:
        raise Exception(f"Failed to create RAG tool: {str(e)}")
