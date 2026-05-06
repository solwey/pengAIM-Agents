"""Direct RAG API client for the Adaptive-CRAG agent.

Unlike the ReAct agent which uses LLM tool-calling to invoke RAG,
this client makes direct HTTP calls controlled by the graph structure.
"""

import logging
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig

from aegra_api.settings import settings
from graphs.react_agent.rag_models import (
    DocumentCollectionInfo,
    RagToolResponse,
    SourceDocument,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = httpx.Timeout(120.0)

EMPTY_RESPONSE = RagToolResponse(context_text="", sources=[], retrieval_metadata={}, document_collections=[])


def _error_response(reason: str) -> RagToolResponse:
    return RagToolResponse(
        context_text="",
        sources=[],
        retrieval_metadata={"error": reason},
        document_collections=[],
    )


def _get_auth(config: RunnableConfig) -> str | None:
    configurable = config.get("configurable", {})
    return configurable.get("langgraph_auth_user", {}).get("authorization")


def _get_rag_base_url(config: RunnableConfig) -> str:
    base = settings.graphs.RAG_API_URL.rstrip("/")
    if "/tenant/" in base:
        return base
    tenant_id = config.get("configurable", {}).get("tenant_id")
    if tenant_id:
        return f"{base}/tenant/{tenant_id}"
    return base


def _get_rag_key_id(config: RunnableConfig) -> str | None:
    configurable = config.get("configurable", {})
    model_name = configurable.get("model_name", "")
    is_google = model_name.lower().startswith("google") or model_name.lower().startswith("gemini")
    if is_google:
        key_data = configurable.get("rag_google_api_key", {})
    else:
        key_data = configurable.get("rag_openai_api_key", {})
    return key_data.get("keyId")


async def search_rag(
    query: str,
    config: RunnableConfig,
    *,
    retrieval_mode: str | None = None,
) -> RagToolResponse:
    """Call RAG API /query/retrieve for semantic graph search."""
    authorization = _get_auth(config)
    if not authorization:
        logger.error("No authorization token found")
        return _error_response("no_auth_token")

    base_url = _get_rag_base_url(config)
    endpoint = f"{base_url}/query/retrieve"

    configurable = config.get("configurable", {})
    model_name = configurable.get("model_name") or "openai:gpt-4o-mini"
    is_google = model_name.lower().startswith("google") or model_name.lower().startswith("gemini")
    llm_model = model_name.split(":", 1)[1] if ":" in model_name else model_name

    body: dict[str, Any] = {
        "question": query,
        "system_prompt": configurable.get("rag_system_prompt"),
        "retrieval_mode": retrieval_mode or configurable.get("rag_retrieval_mode") or "rrf",
        "api_key_id": _get_rag_key_id(config),
        "llm_provider": "gemini" if is_google else "open-ai",
        "llm_model": llm_model,
        "embedding_model": configurable.get("rag_embedding_model") or "text-embedding-3-small",
        "context_token_budget": configurable.get("rag_retrieval_context_token_budget"),
        "text_unit_ratio": configurable.get("rag_retrieval_text_unit_ratio"),
        "community_ratio": configurable.get("rag_retrieval_community_ratio"),
        "entity_ratio": configurable.get("rag_retrieval_entity_ratio"),
        "relationship_ratio": configurable.get("rag_retrieval_relationship_ratio"),
        "top_k_relationships": configurable.get("rag_retrieval_top_k_relationships"),
        "top_k_entities": configurable.get("rag_retrieval_top_k_entities"),
        "chunk_top_k_per_entity": configurable.get("rag_retrieval_chunk_top_k_per_entity"),
        "chunk_ranking_overfetch": configurable.get("rag_retrieval_chunk_ranking_overfetch"),
        "chunk_rank_weight_similarity": configurable.get("rag_retrieval_chunk_rank_weight_similarity"),
        "chunk_rank_weight_entity": configurable.get("rag_retrieval_chunk_rank_weight_entity"),
    }

    headers = {"authorization": authorization, "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(endpoint, json=body, headers=headers)

        if not response.is_success:
            logger.error("RAG API error %d for query '%s': %s", response.status_code, query[:80], response.text[:300])
            return _error_response(f"http_{response.status_code}")

        data = response.json()
        sources = []
        for src in data.get("sources", []):
            try:
                sources.append(SourceDocument(**src))
            except Exception:
                logger.warning("Skipped malformed source: %s", str(src)[:200])
                continue

        collections = []
        for coll in data.get("document_collections", []):
            try:
                collections.append(DocumentCollectionInfo(**coll))
            except Exception:
                logger.warning("Skipped malformed collection: %s", str(coll)[:200])
                continue

        return RagToolResponse(
            context_text=data.get("context_text", ""),
            sources=sources,
            retrieval_metadata=data.get("retrieval_metadata", {}),
            document_collections=collections,
        )
    except httpx.TimeoutException:
        logger.error("RAG API timed out after 120s for query: %s", query[:100])
        return _error_response("timeout")
    except Exception:
        logger.exception("RAG API request failed")
        return _error_response("exception")


async def lookup_documents(
    config: RunnableConfig,
    *,
    query: str | None = None,
    reference_date: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Call /query/documents/lookup for BM25 keyword search and date filtering."""
    authorization = _get_auth(config)
    if not authorization:
        logger.error("No authorization token found")
        return {"error": "no_auth_token", "results": []}

    if not query and not reference_date:
        return {"error": "missing_query_or_date", "results": []}

    base_url = _get_rag_base_url(config)
    endpoint = f"{base_url}/query/documents/lookup"

    body: dict[str, Any] = {"limit": limit}
    if query:
        body["query"] = query
    if reference_date:
        body["reference_date"] = reference_date

    headers = {"authorization": authorization, "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.post(endpoint, json=body, headers=headers)

        if not response.is_success:
            logger.error("Lookup API error %d: %s", response.status_code, response.text[:300])
            return {"error": f"http_{response.status_code}", "results": []}

        data = response.json()
        return {
            "error": "",
            "query": data.get("query"),
            "reference_date": data.get("reference_date"),
            "results": data.get("results", []),
        }
    except httpx.TimeoutException:
        logger.error("Lookup API timed out")
        return {"error": "timeout", "results": []}
    except Exception:
        logger.exception("Lookup API request failed")
        return {"error": "exception", "results": []}


async def get_document_content(
    config: RunnableConfig,
    doc_id: str,
    *,
    page: int = 1,
    page_size: int = 50,
    doc_page: int | None = None,
) -> dict[str, Any]:
    """Call /query/documents/{doc_id}/content for ordered walkthrough of a document."""
    authorization = _get_auth(config)
    if not authorization:
        logger.error("No authorization token found")
        return {"error": "no_auth_token", "items": []}

    base_url = _get_rag_base_url(config)
    endpoint = f"{base_url}/query/documents/{doc_id}/content"

    params: dict[str, Any] = {"page": page, "page_size": page_size}
    if doc_page is not None:
        params["doc_page"] = doc_page

    headers = {"authorization": authorization}

    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            response = await client.get(endpoint, params=params, headers=headers)

        if not response.is_success:
            logger.error("Document content API error %d for doc %s: %s", response.status_code, doc_id, response.text[:300])
            return {"error": f"http_{response.status_code}", "items": []}

        data = response.json()
        return {
            "error": "",
            "file_name": data.get("file_name"),
            "reference_date": data.get("reference_date"),
            "items": data.get("items", []),
            "meta": data.get("meta", {}),
        }
    except httpx.TimeoutException:
        logger.error("Document content API timed out for doc %s", doc_id)
        return {"error": "timeout", "items": []}
    except Exception:
        logger.exception("Document content API request failed")
        return {"error": "exception", "items": []}
