"""Pydantic models for RAG tool structured output."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """Individual source document with metadata.

    Sources can include multiple context types:
    - Text units: Direct document passages
    - Community reports: Aggregated insights across multiple documents
    - Entities: Key concepts, people, organizations
    - Relationships: Connections between entities
    """

    id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Document title or entity name")
    content: str = Field(..., description="Document content snippet or description")
    source_type: str = Field(..., description="Source type: text_unit, community_report, entity, relationship")
    chunk_id: Optional[str] = Field(None, description="Chunk ID for text_unit sources")
    relevance_score: Optional[float] = Field(None, description="Relevance score (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata (entity type, relationship type, etc.)"
    )


class DocumentCollectionInfo(BaseModel):
    """Document with its collection ID mapping.

    Maps document UUIDs to their parent collection IDs, allowing for
    collection-level attribution and organization of retrieved documents.
    """

    document_id: str = Field(..., description="Document UUID")
    collection_id: str = Field(..., description="Collection ID the document belongs to")
    document_title: Optional[str] = Field(None, description="Document title (file name)")


class RagToolResponse(BaseModel):
    """Structured response from RAG tool matching RagContextResponse API.

    This response provides comprehensive context for grounding LLM answers:
    - context_text: Full retrieved context as a single string for direct use
    - sources: Individual source documents with metadata for citation
    - retrieval_metadata: Information about the retrieval process
    - document_collections: Mapping of documents to their collections
    """

    context_text: str = Field(
        ...,
        description="Full retrieved context as a single string"
    )
    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="List of source documents with metadata and relevance scores"
    )
    retrieval_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the retrieval process (mode, timing, counts, etc.)"
    )
    document_collections: List[DocumentCollectionInfo] = Field(
        default_factory=list,
        description="List of documents with their collection mappings"
    )

    def format_for_llm(self) -> str:
        """Format the response for optimal LLM consumption.

        Returns:
            A formatted string with context, sources, and metadata
        """
        # Format sources for readability with type information
        sources_sections = []
        for i, src in enumerate(self.sources):
            # Use the dedicated source_type field
            relevance_str = f"{src.relevance_score:.2f}" if src.relevance_score is not None else "N/A"
            source_text = (
                f"Source {i+1} [{src.source_type}] (relevance: {relevance_str}):\n"
                f"Title: {src.title}\n"
                f"Content: {src.content}"
            )

            # Add additional metadata if present
            if src.metadata and "entity_type" in src.metadata:
                source_text += f"\nEntity Type: {src.metadata['entity_type']}"
            if src.metadata and "relationship_type" in src.metadata:
                source_text += f"\nRelationship Type: {src.metadata['relationship_type']}"

            sources_sections.append(source_text)

        sources_text = "\n\n".join(sources_sections) if sources_sections else "No sources available"

        # Format metadata
        metadata_items = [f"{k}: {v}" for k, v in self.retrieval_metadata.items()]
        metadata_text = "\n".join(metadata_items) if metadata_items else "No metadata available"

        return f"""Retrieved Context:
{self.context_text}

Sources:
{sources_text}

Retrieval Metadata:
{metadata_text}
"""


class RagToolError(BaseModel):
    """Error response from RAG tool with structured error information."""

    error: str = Field(..., description="Human-readable error message")
    error_type: str = Field(
        default="unknown",
        description="Error type (auth_error, network_error, timeout_error, etc.)"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details for debugging"
    )
