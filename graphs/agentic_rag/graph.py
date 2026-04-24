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
from graphs.react_agent.rag_models import DocumentCollectionInfo, SourceDocument
from graphs.react_agent.utils import get_api_key_for_model, get_today_str

logger = logging.getLogger(__name__)

EXACT_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
NON_DATE_LOOKUP_TERMS = {
    "document", "documents", "file", "files", "record", "records",
    "doc", "docs", "added", "uploaded", "created", "modified", "updated",
    "from", "in", "on", "during", "for", "at", "by", "between", "and",
    "show", "list", "what", "which", "were", "was", "are", "is", "the",
    "a", "an", "of", "to",
}
MONTH_TERMS = {
    "january", "jan", "february", "feb", "march", "mar", "april", "apr",
    "may", "june", "jun", "july", "jul", "august", "aug", "september",
    "sep", "sept", "october", "oct", "november", "nov", "december", "dec",
}
PROBLEM_REPORT_RE = re.compile(
    r"\b(broke|broken|failing|failure|fault|error|issue|problem|malfunction|stuck)\b"
    r"|not working|doesn'?t work|can't figure|cannot figure|won'?t start|will not start",
    re.IGNORECASE,
)


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


def _is_exact_reference_date(value: str | None) -> bool:
    return bool(value and EXACT_DATE_RE.fullmatch(value.strip()))


def _is_date_inventory_query(query: str | None) -> bool:
    """Return True when a temporal query has no non-date lookup terms."""
    tokens = re.findall(r"[a-zA-Z0-9]+", str(query or "").lower())
    meaningful = [
        token for token in tokens
        if token not in NON_DATE_LOOKUP_TERMS
        and token not in MONTH_TERMS
        and not token.isdigit()
    ]
    return not meaningful


def _is_problem_report(query: str | None) -> bool:
    return bool(PROBLEM_REPORT_RE.search(str(query or "")))


def _problem_report_note(query: str | None) -> str:
    if not _is_problem_report(query):
        return ""

    return (
        "Answering note: The user is reporting a problem in vague terms. If the "
        "retrieved context does not identify a single cause, do not give generic "
        "repair advice. Use the context to provide the closest related evidence, "
        "checks, hazards, or procedures with citations, clearly marked as related "
        "leads rather than confirmed fixes. Ask for the missing symptoms needed "
        "to narrow the issue.\n\n"
    )


def _source_doc_id(source: SourceDocument) -> str:
    metadata = source.metadata or {}

    for key in ("doc_id", "document_id"):
        value = metadata.get(key)
        if value:
            return str(value)

    document_ids = metadata.get("document_ids")
    if isinstance(document_ids, list):
        for value in document_ids:
            if value:
                return str(value)
    elif document_ids:
        return str(document_ids)

    if UUID_RE.fullmatch(str(source.title or "")):
        return str(source.title)

    return ""


def _format_citation_context(
    sources: list[SourceDocument],
    document_collections: list[DocumentCollectionInfo],
    *,
    max_sources: int = 50,
) -> str:
    """Build an explicit citation map for the answer LLM."""
    title_by_doc_id = {
        collection.document_id: collection.document_title
        for collection in document_collections
        if collection.document_title
    }

    lines: list[str] = []
    seen_tokens: set[str] = set()
    for source in sources:
        if source.source_type != "text_unit" or not source.chunk_id:
            continue

        doc_id = _source_doc_id(source)
        if not doc_id:
            continue

        token = f"[{source.chunk_id}|{doc_id}]"
        if token in seen_tokens:
            continue
        seen_tokens.add(token)

        title = title_by_doc_id.get(doc_id) or source.title or "Untitled document"
        snippet = re.sub(r"\s+", " ", str(source.content or "")).strip()
        if len(snippet) > 240:
            snippet = snippet[:237].rstrip() + "..."
        lines.append(f"- {token} {title}: {snippet}")

        if len(lines) >= max_sources:
            break

    if not lines:
        return ""

    return (
        "Citation tokens available from text-unit sources:\n"
        "Use only these exact tokens for citations. Do not invent or alter tokens.\n"
        + "\n".join(lines)
    )


def _append_citation_context(
    context: str,
    sources: list[SourceDocument],
    document_collections: list[DocumentCollectionInfo],
) -> str:
    citation_context = _format_citation_context(sources, document_collections)
    if not citation_context:
        if len(context) > MAX_CONTEXT_CHARS:
            return context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated to fit token budget]"
        return context

    reserved = len(citation_context) + len("\n\nRetrieved context:\n")
    context_budget = max(1000, MAX_CONTEXT_CHARS - reserved)
    if len(context) > context_budget:
        context = context[:context_budget] + "\n\n[Context truncated to fit token budget]"
        logger.info("Truncated context to %d chars before adding citation map", context_budget)

    return f"{citation_context}\n\nRetrieved context:\n{context}"


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

    if extracted_date:
        extracted_date = str(extracted_date).strip()
        if extracted_date.lower() in ("none", "null", "n/a"):
            extracted_date = ""

    if query_type == "temporal" and extracted_date and not _is_exact_reference_date(extracted_date):
        logger.info("Route query: ignoring non-day temporal extraction '%s'", extracted_date)
        extracted_date = ""

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

    if query_type == "temporal" and not _is_exact_reference_date(extracted_date):
        result = await _search_semantic(state, config, query, retry_count)
        if result.get("search_context"):
            result["search_context"] = (
                "Retrieval note: the question references a broad or relative time period, "
                "so exact-day metadata filtering was not applied. The context below is "
                "from semantic retrieval rather than date-filtered document lookup.\n\n"
                f"{result['search_context']}"
            )
        result["query_type"] = "factual"
        return result

    if query_type in ("metadata", "overview") or (query_type == "temporal" and extracted_date):
        lookup_query = query
        if query_type == "temporal" and _is_date_inventory_query(query):
            lookup_query = ""
        return await _search_lookup(state, config, lookup_query, extracted_date)

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

    context = _append_citation_context(
        rag_response.context_text,
        rag_response.sources,
        rag_response.document_collections,
    )

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
    document_collections: list[DocumentCollectionInfo] = []
    seen_doc_ids: set[str] = set()
    context_parts: list[str] = []

    for item in items:
        doc_id = str(item.get("doc_id") or "")
        file_name = str(item.get("file_name") or "Untitled")
        collection_id = str(item.get("collection_id") or "")
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

        if doc_id and collection_id and doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            document_collections.append(DocumentCollectionInfo(
                document_id=doc_id,
                collection_id=collection_id,
                document_title=file_name,
                relevance_score=float(score) if score is not None else None,
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
        "document_collections": document_collections,
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
    collection_id = str(target.get("collection_id") or "")

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
        if chunk_id and doc_id:
            parts.append(f"[{chunk_id}|{doc_id}] {text}")
        else:
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

    document_collections = []
    if doc_id and collection_id:
        document_collections.append(DocumentCollectionInfo(
            document_id=doc_id,
            collection_id=collection_id,
            document_title=file_name,
            relevance_score=float(target_score) if target_score is not None else None,
            last_human_message_id=last_human_id,
        ))

    return {
        "search_context": context,
        "sources": sources,
        "document_collections": document_collections,
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

    is_bm25 = query_type in ("metadata", "overview", "temporal")
    threshold = QUALITY_THRESHOLD_BM25 if is_bm25 else QUALITY_THRESHOLD_SEMANTIC

    if not is_bm25 and len(context.strip()) < 200:
        logger.info("Quality gate: FAIL - context too short (%d chars)", len(context.strip()))
        return {"quality_ok": False}

    score_sources = sources if is_bm25 else [s for s in sources if s.source_type == "text_unit"]
    scores = [s.relevance_score for s in score_sources if s.relevance_score is not None]
    if not scores:
        score_scope = "document_lookup sources" if is_bm25 else "text_unit sources"
        logger.info("Quality gate: FAIL - no relevance scores available on %s", score_scope)
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
    problem_note = _problem_report_note(question)

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
                f"{problem_note}"
                "The search returned limited or low-quality results. "
                "Answer as best you can using the available context below, "
                "but clearly state what information is uncertain or missing.\n\n"
                f"Available context:\n{context}"
            )
    else:
        user_content = (
            f"Question: {question}\n\n"
            f"{problem_note}"
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
