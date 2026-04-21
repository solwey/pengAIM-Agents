"""State definitions for the Adaptive-CRAG agent."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from graphs.react_agent.rag_models import DocumentCollectionInfo, SourceDocument


def merge_sources(existing: list[SourceDocument], incoming: list[SourceDocument]) -> list[SourceDocument]:
    if incoming is None:
        return existing
    return incoming


def merge_document_collections(
    existing: list[DocumentCollectionInfo],
    incoming: list[DocumentCollectionInfo],
) -> list[DocumentCollectionInfo]:
    if incoming is None:
        return existing
    return incoming


class AgentInputState(TypedDict):
    messages: list[AnyMessage]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    sources: Annotated[list[SourceDocument], merge_sources]
    document_collections: Annotated[list[DocumentCollectionInfo], merge_document_collections]
    query_type: str
    search_query: str
    extracted_date: str
    search_context: str
    retrieval_error: str
    retry_count: int
    quality_ok: bool


class AgentOutputState(TypedDict):
    messages: list[AnyMessage]
    sources: list[SourceDocument]
    document_collections: list[DocumentCollectionInfo]
