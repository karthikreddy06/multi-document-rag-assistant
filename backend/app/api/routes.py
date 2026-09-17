"""
FastAPI HTTP API Routes for Multi-Document RAG Assistant.
Reuses existing RAGApplication orchestrator, retriever, vector store, and generation services.
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.main import RAGApplication
from app.utils.logger import setup_logger
from app.api.schemas import (
    HealthResponse,
    DocumentsResponse,
    DocumentInfo,
    ChatRequest,
    ChatResponse,
    SourceChunk,
    IngestResponse,
    ReindexResponse,
)

logger = setup_logger("api.routes")

# Initialize FastAPI application
app = FastAPI(
    title="Multi-Document RAG Assistant API",
    description="REST API layer for multi-document retrieval-augmented generation and ingestion.",
    version="1.0.0",
)

# Configure CORS for local development and frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton container for RAGApplication
_rag_app_instance: Optional[RAGApplication] = None


def get_rag_app() -> RAGApplication:
    """
    Dependency provider for RAGApplication.
    Instantiates lazily on first request to prevent blocking module import.
    """
    global _rag_app_instance
    if _rag_app_instance is None:
        logger.info("Initializing shared RAGApplication instance for API layer...")
        _rag_app_instance = RAGApplication(auto_ingest=False)
    return _rag_app_instance


# ------------------------------------------------------------------------------
# 1. Health Endpoint
# ------------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(status="ok")


# ------------------------------------------------------------------------------
# 2. Documents Metadata Endpoint
# ------------------------------------------------------------------------------
@app.get("/api/documents", response_model=DocumentsResponse, tags=["Documents"])
def get_documents(rag_app: RAGApplication = Depends(get_rag_app)) -> DocumentsResponse:
    """
    Return currently indexed documents and existing metadata from ChromaDB.
    """
    try:
        indexed_map = rag_app.vector_store.get_indexed_files()
        total_chunks = rag_app.vector_store.count()

        if total_chunks == 0:
            return DocumentsResponse(
                total_documents=0,
                total_chunks=0,
                documents=[],
            )

        # Retrieve all chunk metadata from the active collection
        raw = rag_app.vector_store.collection.get(include=["metadatas"])
        metadatas = raw.get("metadatas", []) or []

        # Aggregate per-document statistics
        doc_stats: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            fn = meta.get("filename", "unknown")
            if fn not in doc_stats:
                doc_stats[fn] = {
                    "filename": fn,
                    "file_hash": meta.get("file_hash") or indexed_map.get(fn),
                    "chunk_count": 0,
                    "pages": set(),
                    "sections": set(),
                }
            doc_stats[fn]["chunk_count"] += 1
            if meta.get("page_number"):
                doc_stats[fn]["pages"].add(meta.get("page_number"))
            sec = meta.get("section")
            if sec and sec != "General":
                doc_stats[fn]["sections"].add(sec)

        doc_list: List[DocumentInfo] = []
        for fn, info in sorted(doc_stats.items(), key=lambda x: x[0]):
            doc_list.append(
                DocumentInfo(
                    filename=info["filename"],
                    file_hash=info["file_hash"],
                    chunk_count=info["chunk_count"],
                    page_count=len(info["pages"]) if info["pages"] else None,
                    sections=sorted(list(info["sections"]))[:10],
                )
            )

        return DocumentsResponse(
            total_documents=len(doc_list),
            total_chunks=total_chunks,
            documents=doc_list,
        )
    except Exception as e:
        logger.error(f"Error retrieving documents metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document metadata from vector store."
        )


# ------------------------------------------------------------------------------
# 3. Chat Endpoint
# ------------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
def chat(
    request: ChatRequest,
    rag_app: RAGApplication = Depends(get_rag_app),
) -> ChatResponse:
    """
    Execute grounded RAG query through existing retriever and LLM generator.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string must not be empty.",
        )

    try:
        # 1. Retrieve relevant chunks using existing HybridRetriever
        chunks = rag_app.retriever.retrieve(query)

        # 2. Generate grounded answer using existing LLMGenerator
        answer = rag_app.generator.generate_answer(query, chunks)

        # 3. Format source chunk provenance
        sources: List[SourceChunk] = []
        for chunk in chunks:
            sources.append(
                SourceChunk(
                    filename=str(chunk.metadata.get("filename", "unknown")),
                    page=chunk.metadata.get("page_number"),
                    section=chunk.metadata.get("section"),
                    chunk_id=chunk.metadata.get("chunk_id"),
                    score=round(chunk.score, 4) if chunk.score is not None else None,
                    text=chunk.text[:300] if request.show_context else None,
                )
            )

        return ChatResponse(answer=answer, sources=sources)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat query '{query}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer from documents.",
        )


# ------------------------------------------------------------------------------
# 4. Ingest Endpoint
# ------------------------------------------------------------------------------
@app.post("/api/ingest", response_model=IngestResponse, tags=["Ingestion"])
def trigger_ingestion(rag_app: RAGApplication = Depends(get_rag_app)) -> IngestResponse:
    """
    Trigger incremental document ingestion on the documents directory.
    """
    try:
        res = rag_app.pipeline.ingest_directory(force=False)
        docs_count = res.get("documents", 0)
        chunks_count = res.get("chunks", 0)
        return IngestResponse(
            status="success",
            documents_processed=docs_count,
            chunks_stored=chunks_count,
            message=f"Ingestion completed. Processed {docs_count} documents, stored {chunks_count} chunks.",
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed.",
        )


# ------------------------------------------------------------------------------
# 5. Reindex Endpoint
# ------------------------------------------------------------------------------
@app.post("/api/reindex", response_model=ReindexResponse, tags=["Ingestion"])
def trigger_reindex(rag_app: RAGApplication = Depends(get_rag_app)) -> ReindexResponse:
    """
    Clear vector store and re-index all documents from scratch.
    """
    try:
        rag_app.vector_store.clear()
        res = rag_app.pipeline.ingest_directory(force=True)
        docs_count = res.get("documents", 0)
        chunks_count = res.get("chunks", 0)
        return ReindexResponse(
            status="success",
            documents_processed=docs_count,
            chunks_stored=chunks_count,
            message=f"Reindexing completed. Cleared database and indexed {chunks_count} chunks from {docs_count} documents.",
        )
    except Exception as e:
        logger.error(f"Reindexing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document reindexing failed.",
        )
