"""Adaptive-CRAG agent graph.

Flow:
  START → route_query → search → quality_gate →
    [quality_ok=True]  → generate_answer → END
    [quality_ok=False, retry < MAX] → rewrite_query → search → quality_gate → ...
    [quality_ok=False, retry >= MAX] → generate_answer → END
"""

import json
import logging
import re
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from aegra_api.settings import settings
from graphs.agentic_rag.prompts import (
    GENERATE_SYSTEM_PROMPT,
    MAX_CONTEXT_CHARS,
    MAX_RETRIES,
    OFF_TOPIC_SYSTEM_PROMPT,
    QUALITY_THRESHOLD_BM25,
    QUALITY_THRESHOLD_SEMANTIC,
    REWRITE_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    WALKTHROUGH_MIN_SCORE,
    WALKTHROUGH_PAGE_SIZE,
)
from graphs.agentic_rag.rag_client import (
    get_document_content,
    lookup_documents,
    search_rag,
)
from graphs.agentic_rag.state import (
    AgentInputState,
    AgentOutputState,
    AgentState,
)
from graphs.react_agent.context import Context
from graphs.react_agent.rag_models import SourceDocument
from graphs.react_agent.utils import get_api_key_for_model, get_today_str

logger = logging.getLogger(__name__)


def _get_last_human_message(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _get_last_human_message_id(state: AgentState) -> str | None:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return getattr(msg, "id", None)
    return None


async def _get_model(config: RunnableConfig) -> Any:
    cfg = Context(**config.get("configurable", {}))
    api_key = await get_api_key_for_model(cfg.model_name or "", config)
    model_name = cfg.model_name
    if settings.graphs.AZURE_OPENAI_ENDPOINT:
        model_name = re.sub(r"^openai:", "azure_openai:", model_name)
    return init_chat_model(
        model_name,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        api_key=api_key or "No token found",
    )


# ── Node 1: Route Query ──────────────────────────────────────────────────────

async def route_query(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Classify the user's question and produce an optimized search query."""
    question = _get_last_human_message(state)
    if not question:
        return {
            "query_type": "off_topic",
            "search_query": "",
            "extracted_date": "",
            "search_context": "",
            "retrieval_error": "",
            "retry_count": 0,
            "quality_ok": False,
            "sources": [],
            "document_collections": [],
        }

    model = await _get_model(config)
    response = await model.ainvoke([
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ])

    content = response.content if isinstance(response.content, str) else str(response.content)

    try:
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0]
        parsed = json.loads(clean)
        query_type = parsed.get("query_type", "factual")
        search_query = parsed.get("search_query", question)
        extracted_date = parsed.get("extracted_date") or ""
    except (json.JSONDecodeError, KeyError):
        logger.warning("Router failed to parse LLM output, defaulting to factual: %s", content[:200])
        query_type = "factual"
        search_query = question
        extracted_date = ""

    if query_type not in ("factual", "metadata", "temporal", "walkthrough", "overview", "off_topic"):
        query_type = "factual"

    logger.info("Route query: type=%s, date=%s, search_query=%s", query_type, extracted_date, search_query[:100])

    return {
        "query_type": query_type,
        "search_query": search_query,
        "extracted_date": extracted_date,
        "retry_count": 0,
        "quality_ok": False,
        "sources": [],
        "document_collections": [],
    }


# ── Node 2: Search ───────────────────────────────────────────────────────────

async def search(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Execute the appropriate retrieval path based on query type."""
    query = state.get("search_query", "")
    query_type = state.get("query_type", "factual")
    extracted_date = state.get("extracted_date", "") or None
    retry_count = state.get("retry_count", 0)

    if query_type == "off_topic":
        return {"search_context": "", "sources": [], "document_collections": [], "retrieval_error": ""}

    if query_type == "temporal" and not extracted_date:
        result = await _search_semantic(state, config, query, retry_count)
        result["query_type"] = "factual"
        return result

    if query_type in ("metadata", "overview") or (query_type == "temporal" and extracted_date):
        return await _search_lookup(state, config, query, extracted_date)

    if query_type == "walkthrough":
        return await _search_walkthrough(state, config, query)

    return await _search_semantic(state, config, query, retry_count)


async def _search_semantic(
    state: AgentState,
    config: RunnableConfig,
    query: str,
    retry_count: int,
) -> dict[str, Any]:
    """Semantic GraphRAG search via /query/retrieve."""
    configurable = config.get("configurable", {})
    if retry_count > 0:
        retrieval_mode = "rrf" if retry_count == 1 else "hyde"
    else:
        retrieval_mode = configurable.get("rag_retrieval_mode")

    rag_response = await search_rag(query, config, retrieval_mode=retrieval_mode)

    last_human_id = _get_last_human_message_id(state)
    for source in rag_response.sources:
        source.last_human_message_id = last_human_id
    for coll in rag_response.document_collections:
        coll.last_human_message_id = last_human_id

    context = rag_response.context_text
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated to fit token budget]"
        logger.info("Truncated context from %d to %d chars", len(rag_response.context_text), MAX_CONTEXT_CHARS)

    retrieval_error = ""
    if isinstance(rag_response.retrieval_metadata, dict):
        retrieval_error = rag_response.retrieval_metadata.get("error", "") or ""

    return {
        "search_context": context,
        "sources": rag_response.sources,
        "document_collections": rag_response.document_collections,
        "retrieval_error": retrieval_error,
    }


async def _search_lookup(
    state: AgentState,
    config: RunnableConfig,
    query: str,
    reference_date: str | None,
) -> dict[str, Any]:
    """BM25 keyword + date lookup via /query/documents/lookup."""
    effective_query = query or None
    if not effective_query and not reference_date:
        effective_query = _get_last_human_message(state) or None

    result = await lookup_documents(
        config,
        query=effective_query,
        reference_date=reference_date,
        limit=10,
    )

    if result.get("error"):
        fallback = await _search_semantic(state, config, query or _get_last_human_message(state), retry_count=0)
        if fallback.get("search_context"):
            fallback["query_type"] = "factual"
            return fallback
        return {
            "search_context": "",
            "sources": [],
            "document_collections": [],
            "retrieval_error": result["error"],
        }

    items = result.get("results", [])
    if not items:
        return {
            "search_context": "",
            "sources": [],
            "document_collections": [],
            "retrieval_error": "",
        }

    last_human_id = _get_last_human_message_id(state)
    sources: list[SourceDocument] = []
    context_parts: list[str] = []

    for item in items:
        doc_id = str(item.get("doc_id") or "")
        file_name = str(item.get("file_name") or "Untitled")
        ref_date = item.get("reference_date") or ""
        match_type = str(item.get("match_type") or "keyword")
        score = item.get("score")

        sources.append(SourceDocument(
            id=doc_id,
            title=file_name,
            content=f"Document match via {match_type}" + (f" (date: {ref_date})" if ref_date else ""),
            source_type="document_lookup",
            relevance_score=float(score) if score is not None else None,
            metadata={"match_type": match_type, "reference_date": ref_date, "doc_id": doc_id},
            last_human_message_id=last_human_id,
        ))

        line = f"- {file_name}"
        if ref_date:
            line += f" (date: {ref_date})"
        if score is not None:
            line += f" [score: {float(score):.3f}]"
        context_parts.append(line)

    context = "Matching documents:\n" + "\n".join(context_parts)

    return {
        "search_context": context,
        "sources": sources,
        "document_collections": [],
        "retrieval_error": "",
    }


async def _search_walkthrough(
    state: AgentState,
    config: RunnableConfig,
    query: str,
) -> dict[str, Any]:
    """Walkthrough: look up document by name, then fetch ordered chunks."""
    lookup = await lookup_documents(config, query=query, limit=3)

    if lookup.get("error"):
        fallback = await _search_semantic(state, config, query, retry_count=0)
        if fallback.get("search_context"):
            fallback["query_type"] = "factual"
            return fallback
        return {
            "search_context": "",
            "sources": [],
            "document_collections": [],
            "retrieval_error": lookup["error"],
        }

    results = lookup.get("results", [])
    if not results:
        return {
            "search_context": "",
            "sources": [],
            "document_collections": [],
            "retrieval_error": "",
        }

    target = results[0]
    target_score = target.get("score")
    if target_score is not None and float(target_score) < WALKTHROUGH_MIN_SCORE:
        logger.info("Walkthrough: top match score %.2f below %.2f, falling back to semantic", float(target_score), WALKTHROUGH_MIN_SCORE)
        fallback = await _search_semantic(state, config, query, retry_count=0)
        if fallback.get("search_context"):
            fallback["query_type"] = "factual"
            return fallback
        return {
            "search_context": "",
            "sources": [],
            "document_collections": [],
            "retrieval_error": "",
        }

    doc_id = str(target.get("doc_id") or "")
    file_name = str(target.get("file_name") or "Untitled")

    content = await get_document_content(config, doc_id, page=1, page_size=WALKTHROUGH_PAGE_SIZE)

    if content.get("error"):
        return {
            "search_context": "",
            "sources": [],
            "document_collections": [],
            "retrieval_error": content["error"],
        }

    items = content.get("items", [])
    if not items:
        return {
            "search_context": "",
            "sources": [],
            "document_collections": [],
            "retrieval_error": "",
        }

    last_human_id = _get_last_human_message_id(state)
    sources: list[SourceDocument] = []
    parts: list[str] = [f"Document: {file_name}"]

    for item in items:
        chunk_id = str(item.get("chunk_id") or "")
        order_index = item.get("order_index")
        page_number = item.get("page_number")
        section = item.get("section_title") or ""
        text = item.get("text") or ""

        header = ""
        if section:
            header = f"[{section}]"
        if page_number is not None:
            header = (header + f" (page {page_number})").strip()
        if header:
            parts.append(header)
        parts.append(text)

        sources.append(SourceDocument(
            id=chunk_id,
            title=f"{file_name} - chunk {order_index if order_index is not None else chunk_id}",
            content=text[:500],
            source_type="text_unit",
            chunk_id=chunk_id,
            metadata={
                "doc_id": doc_id,
                "order_index": order_index,
                "page_number": page_number,
                "section_title": section,
            },
            last_human_message_id=last_human_id,
        ))

    context = "\n\n".join(parts)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n\n[Walkthrough truncated - request next page for more]"

    return {
        "search_context": context,
        "sources": sources,
        "document_collections": [],
        "retrieval_error": "",
    }


# ── Node 3: Quality Gate ─────────────────────────────────────────────────────

async def quality_gate(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Check if search results meet quality threshold."""
    context = state.get("search_context", "")
    sources: list[SourceDocument] = state.get("sources", [])
    query_type = state.get("query_type", "factual")

    if query_type == "off_topic":
        return {"quality_ok": True}

    if query_type == "walkthrough":
        if context and sources:
            logger.info("Quality gate: PASS (walkthrough) - %d chunks", len(sources))
            return {"quality_ok": True}
        logger.info("Quality gate: FAIL (walkthrough) - no content")
        return {"quality_ok": False}

    if not context or not context.strip():
        logger.info("Quality gate: FAIL - empty context")
        return {"quality_ok": False}

    if not sources:
        logger.info("Quality gate: FAIL - no sources")
        return {"quality_ok": False}

    if len(context.strip()) < 200:
        logger.info("Quality gate: FAIL - context too short (%d chars)", len(context.strip()))
        return {"quality_ok": False}

    is_bm25 = query_type in ("metadata", "overview", "temporal")
    threshold = QUALITY_THRESHOLD_BM25 if is_bm25 else QUALITY_THRESHOLD_SEMANTIC

    scores = [s.relevance_score for s in sources if s.relevance_score is not None]
    if not scores:
        logger.info("Quality gate: FAIL - no relevance scores available on any source")
        return {"quality_ok": False}

    best_score = max(scores)
    if best_score < threshold:
        logger.info("Quality gate: FAIL - best score %.2f < threshold %.2f (%s)", best_score, threshold, "bm25" if is_bm25 else "semantic")
        return {"quality_ok": False}

    logger.info("Quality gate: PASS - %d sources, context length %d", len(sources), len(context))
    return {"quality_ok": True}


# ── Node 4: Rewrite Query ────────────────────────────────────────────────────

async def rewrite_query(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Rewrite the search query for a retry attempt."""
    retry_count = state.get("retry_count", 0) + 1
    original_query = state.get("search_query", "")
    query_type = state.get("query_type", "factual")

    if retry_count >= MAX_RETRIES:
        logger.info("Rewrite query (retry %d): skipping rewrite before HYDE mode", retry_count)
        return {
            "retry_count": retry_count,
            "search_context": "",
        }

    model = await _get_model(config)
    prompt = REWRITE_SYSTEM_PROMPT.format(original_query=original_query, query_type=query_type)

    response = await model.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Rewrite this query for better search results: {original_query}"),
    ])

    new_query = response.content if isinstance(response.content, str) else str(response.content)
    new_query = new_query.strip().strip('"').strip("'")

    logger.info("Rewrite query (retry %d): '%s' → '%s'", retry_count, original_query[:80], new_query[:80])

    return {
        "search_query": new_query,
        "retry_count": retry_count,
        "search_context": "",
    }


# ── Node 5: Generate Answer ──────────────────────────────────────────────────

async def generate_answer(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Generate final answer using retrieved context."""
    context = state.get("search_context", "")
    query_type = state.get("query_type", "factual")
    quality_ok = state.get("quality_ok", False)
    question = _get_last_human_message(state)

    cfg = Context(**config.get("configurable", {}))
    model = await _get_model(config)

    user_system_prompt = (cfg.system_prompt or "").strip()
    if query_type == "off_topic":
        technical_prompt = OFF_TOPIC_SYSTEM_PROMPT.format(date=get_today_str())
    else:
        technical_prompt = GENERATE_SYSTEM_PROMPT.format(date=get_today_str())
    system = f"{user_system_prompt}\n\n{technical_prompt}" if user_system_prompt else technical_prompt

    retrieval_error = state.get("retrieval_error", "")

    if query_type == "off_topic":
        user_content = question
    elif retrieval_error and not context:
        user_content = (
            f"Question: {question}\n\n"
            f"The knowledge base search failed with error: {retrieval_error}. "
            "Inform the user that the search integration is temporarily unavailable "
            "and suggest they try again shortly. Do not make up an answer."
        )
    elif not context or not quality_ok:
        if not context:
            user_content = (
                f"Question: {question}\n\n"
                "No relevant information was found in the knowledge base after searching. "
                "Inform the user that the knowledge base does not contain enough information "
                "to answer this question. Do not make up an answer."
            )
        else:
            user_content = (
                f"Question: {question}\n\n"
                "The search returned limited or low-quality results. "
                "Answer as best you can using the available context below, "
                "but clearly state what information is uncertain or missing.\n\n"
                f"Available context:\n{context}"
            )
    else:
        user_content = (
            f"Question: {question}\n\n"
            f"Retrieved context from knowledge base:\n{context}"
        )

    response = await model.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=user_content),
    ])

    if not isinstance(response, AIMessage):
        response = AIMessage(content=str(response.content))

    return {"messages": [response]}


# ── Routing Functions ─────────────────────────────────────────────────────────

def route_after_quality_gate(state: AgentState) -> Literal["generate_answer", "rewrite_query"]:
    """Route based on quality gate result and retry count."""
    quality_ok = state.get("quality_ok", False)
    retry_count = state.get("retry_count", 0)
    query_type = state.get("query_type", "factual")

    if quality_ok or query_type == "off_topic":
        return "generate_answer"

    if retry_count >= MAX_RETRIES:
        logger.info("Max retries (%d) reached, generating answer with available context", MAX_RETRIES)
        return "generate_answer"

    return "rewrite_query"


def route_after_router(state: AgentState) -> Literal["search", "generate_answer"]:
    """Skip search for off-topic queries."""
    if state.get("query_type") == "off_topic":
        return "generate_answer"
    return "search"


# ── Build Graph ───────────────────────────────────────────────────────────────

builder = StateGraph(
    AgentState,
    input=AgentInputState,
    output=AgentOutputState,
    config_schema=Context,
)

builder.add_node("route_query", route_query)
builder.add_node("search", search)
builder.add_node("quality_gate", quality_gate)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "route_query")
builder.add_conditional_edges("route_query", route_after_router, {"search": "search", "generate_answer": "generate_answer"})
builder.add_edge("search", "quality_gate")
builder.add_conditional_edges("quality_gate", route_after_quality_gate, {"generate_answer": "generate_answer", "rewrite_query": "rewrite_query"})
builder.add_edge("rewrite_query", "search")
builder.add_edge("generate_answer", END)

graph = builder.compile()
