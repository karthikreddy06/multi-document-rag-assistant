"""
Main Application Entrypoint for Production RAG System.
Supports interactive question answering, one-shot CLI execution, and document ingestion.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from app.config import settings
from app.embeddings.service import EmbeddingService
from app.generation.generator import LLMGenerator
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.retriever import HybridRetriever
from app.utils.logger import setup_logger
from app.vectorstore.store import VectorStore

logger = setup_logger("app.main")


class RAGApplication:
    """Production RAG orchestrator linking ingestion, retrieval, and generation."""

    def __init__(self, auto_ingest: bool = True):
        self.vector_store = VectorStore()
        self.embedding_service = EmbeddingService(
            model=settings.effective_embedding_model,
            provider=settings.embedding_provider,
            dimensions=settings.effective_embedding_dimensions,
        )
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            top_k=settings.top_k,
        )
        self.generator = LLMGenerator(
            model=settings.effective_llm_model,
            host=settings.ollama_host,
            provider=settings.llm_provider,
            api_url=settings.effective_llm_api_url,
            api_key=settings.effective_llm_api_key,
        )
        self.pipeline = IngestionPipeline(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
        )
        if auto_ingest:
            self.auto_ingest()

    def check_prerequisites(self) -> bool:
        """Check database and AI service provider availability."""
        if not self.embedding_service.check_health():
            if self.embedding_service.provider == "ollama":
                print("\n[WARNING] Could not verify Ollama connection or required models.")
                print("Ensure Ollama is running ('ollama serve') and models are pulled:\n")
                print(f"  ollama pull {settings.embedding_model}")
                print(f"  ollama pull {settings.llm_model}\n")
            elif self.embedding_service.provider in ("local", "onnx"):
                print(f"\n[WARNING] Could not initialize local CPU embedding model '{settings.effective_embedding_model}'.")
            else:
                print(f"\n[WARNING] Could not verify cloud embedding service at {settings.effective_embedding_api_url}.")
            return False
        return True

    def auto_ingest(self) -> None:
        """Scan documents folder and incrementally ingest any new or modified PDFs."""
        print(f"\n[Startup Ingestion] Scanning documents from {settings.documents_abs_path}...")
        res = self.pipeline.ingest_directory(force=False)
        print(
            f"[Startup Ingestion] Complete! Processed: {res.get('documents', 0)} documents, "
            f"Total chunks in database: {self.vector_store.count()}"
        )

    def ingest(self, force: bool = False) -> None:
        """Run document ingestion pipeline."""
        print(f"\n[Ingestion] Scanning documents from {settings.documents_abs_path}...")
        res = self.pipeline.ingest_directory(force=force)
        print(f"[Ingestion] Complete! Documents processed: {res.get('documents', 0)}, Chunks stored: {res.get('chunks', 0)}")

    def answer_question(self, question: str, show_context: bool = False) -> str:
        """Execute the end-to-end RAG query flow."""
        if not question or not question.strip():
            return "Please provide a non-empty question."

        if hasattr(self.retriever, "retrieve_adaptive"):
            chunks, plan, _ = self.retriever.retrieve_adaptive(question)
            budget = getattr(plan, "generation_budget", None)
            num_pred = getattr(budget, "num_predict", None)
            num_ctx = getattr(budget, "num_ctx", None)
        else:
            chunks = self.retriever.retrieve(question)
            plan, num_pred, num_ctx = None, None, None

        if show_context and chunks:
            print("\n--- Retrieved Context Chunks ---")
            for i, chunk in enumerate(chunks, 1):
                print(f"[{i}] Section: {chunk.section} (Score: {chunk.score:.3f})")
                snippet = chunk.text[:200].encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
                print(f"{snippet}...\n")
            print("--- End Context ---\n")

        answer = self.generator.generate_answer(
            question=question,
            chunks=chunks,
            plan=plan,
            num_predict=num_pred,
            num_ctx=num_ctx,
        )
        return answer

    def interactive_loop(self) -> None:
        """Run interactive CLI chat session."""
        print("=" * 60)
        print(" Multi-Document RAG Knowledge Assistant")
        print(" Ask questions about the indexed documents.")
        print(" Type 'exit' or 'quit' to exit.")
        print("=" * 60)

        while True:
            try:
                question = input("\nAsk a question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Goodbye!")
                break

            if not question:
                continue

            if question.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            try:
                answer = self.answer_question(question)
                print("\nAnswer:")
                print(answer)
            except Exception as e:
                print(f"\n[Error] {e}")


def main():
    """CLI Argument parser and dispatcher."""
    parser = argparse.ArgumentParser(description="Production RAG Application")
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single question and output the answer directly."
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run document ingestion on the documents folder."
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Clear existing collection and re-index all documents from scratch."
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved context chunks alongside the answer."
    )

    args = parser.parse_args()

    if args.reindex:
        app = RAGApplication(auto_ingest=False)
        app.vector_store.clear()
        app.ingest(force=True)
        return

    if args.ingest:
        app = RAGApplication(auto_ingest=False)
        app.ingest(force=True)
        return

    app = RAGApplication(auto_ingest=True)

    if args.query:
        ans = app.answer_question(args.query, show_context=args.show_context)
        print(ans)
        return

    # Default: Interactive mode
    app.interactive_loop()


def __getattr__(name: str):
    """
    Expose FastAPI 'app' from app.api.routes dynamically so
    'uvicorn app.main:app' starts the production REST server cleanly.
    """
    if name == "app":
        from app.api.routes import app as fastapi_app
        return fastapi_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
